// =============================================================================
//  pellet_pd.cpp  (v2)  --  Peridynamic thermal cracking of UO2 fuel pellets
//                           and pellet stacks.  Single file, C++17 + OpenMP.
// -----------------------------------------------------------------------------
//  MODEL
//   * 3D ordinary state-based peridynamics, Linear Peridynamic Solid (LPS).
//   * Thermal expansion in the bond extension state:
//         e = |Y| - |X| (1 + alpha (T_bond - Tref)),  T_bond = (T_i + T_j)/2
//   * Optional viscoelastic creep (Mitchell 2011 viscoelastic LPS): deviatoric
//     back-extension e_b relaxes towards e_d with tau(T) (Arrhenius):
//         t_d = 15 mu/m [ e_d - lambda e_b ],  de_b/dt = (e_d - e_b)/tau(T)
//         tau(T) = tauRef exp( Q/Rg (1/T - 1/TtauRef) )
//     integrated semi-implicitly per load step (unconditionally stable).  The
//     hot core relaxes at power; on cool-down it goes into tension -> centre /
//     circumferential cracks.  Without creep a cooled cracked pellet just
//     unloads elastically.
//   * Fracture: bond breaks when its stress-carrying stretch exceeds s_c.
//     s_c from: G_c (Foster et al. 2011 LPS calibration, default), or --sc, or
//     --sigmaf (s_c = sigma_f / E).  --calib-delta sets the horizon so that the
//     G_c- and strength-based values agree (delta = material length).
//     Weibull scatter of s_c: i.i.d. per bond (--wcorr 0) or a MESH-
//     INDEPENDENT grain field of size --wcorr (needed for mesh convergence).
//   * Broken bonds still resist compression (crack-face contact).
//   * Stacks (--stack 0.5,1,0.5): separate bodies, inter-body contact with
//     penalty normal stiffness and Coulomb friction (return mapping, small-
//     sliding column-aligned contact pairs), bottom face pinned, nominal axial
//     pressure on the top face, contact area limited by --contact-rmin and by
//     end dishes/chamfers.  A cracked pellet can be restored into any body
//     from a restart file (--restart, --restart-body).
//   * Thermal field prescribed: T(r) = Ts(q) + q/(4 pi k) (1 - r^2/R^2),
//     Ts = Tref + gsurf q, from a power history (--history t_h q_Wpm [p_Pa]).
//   * Quasi-static per load step: adaptive dynamic relaxation (max-norm
//     residual) + sequential breaking of the most critical bonds
//     (ratio >= kappa * max ratio) with re-relaxation until no bond exceeds s_c.
//
//  OUTPUT (all in --out; designed for re-plotting without re-running)
//   manifest.json       every input + derived quantity, schedule, counts, build
//   history.csv         one row per load step, ~40 columns (loads, solver,
//                       crack area total/per body, damage, crack counts,
//                       stress percentiles, dimensional changes, contact)
//   cycles.csv          every break/relax cycle (step, ratio, bonds, iters)
//   fragments.csv       every fragment at every output step (volume, d_eq,
//                       centroid, extent) per body
//   profile_XXXX.csv    mid-plane radial profiles per body (all stress and
//                       strain components, T, u_r, damage; analytic elastic)
//   angular_XXXX.csv    damage vs angle: mid-plane ring + both end faces, per body
//   interface_XXXX.csv  every contact pair (stacks): gap, pressure, shear, slip
//   state_XXXX.bin      full particle state for crackmesh.py (lattice indices,
//                       fields, broken-neighbour masks)
//   pellet_XXXX.vtk     ParaView point cloud (optional)
//   summary.json        final + peak metrics, fragment statistics, timings
//   restart.bin         (--restart-out) state of a single pellet for reuse
//
//  CLOUD   --ckpt-every n writes <out>/checkpoint.bin atomically; --resume continues from it.
//          SIGTERM (Spot preemption) aborts the current step and exits with code 75.
//
//  BUILD   g++ -O3 -march=native -fopenmp -std=c++17 pellet_pd.cpp -o pellet_pd
//  HELP    ./pellet_pd --help
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <csignal>

#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
static inline int omp_get_max_threads() { return 1; }
static inline double omp_get_wtime() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
#endif

static const char* PD_VERSION = "pellet_pd 2.0";
#ifndef PD_BUILD
#define PD_BUILD "unknown"
#endif
static const char* PD_BUILD_ID = PD_BUILD;   // set by build.sh: -DPD_BUILD=\"<sha256 of source>\"
static volatile std::sig_atomic_t gStop = 0;   // set by SIGTERM/SIGINT (Spot preemption)
static void onSignal(int) { gStop = 1; }
static constexpr int MAXB = 8;               // max bodies in a stack
static constexpr double RGAS = 8.314462618;  // J/mol/K

// ----------------------------------------------------------------------------
// Parameters (SI units; time in hours in histories)
// ----------------------------------------------------------------------------
struct Params {
    // geometry / discretisation
    double R = 4.1e-3, H = 9.8e-3;
    double dishR = 0, dishD = 0, chamfer = 0;  // end dish radius/depth, 45deg chamfer width
    int nr = 16;                                // particles across radius
    double mfac = 3.015;                        // delta / dx (if --delta not given)
    double delta = 0;                           // >0: fixed physical horizon [m]
    double jitter = 0;                          // lattice jitter, fraction of dx (<=0.3)
    std::string stack = "1";                    // body heights as multiples of H
    // material
    double E = 200e9, nu = 0.32, alpha = 1.0e-5, Gc = 10.0;
    double sc = 0, sigmaf = 0;                  // direct critical stretch / tensile strength
    bool calibDelta = false;                    // delta from Gc and sigmaf
    double weibull = 12.0, wcorr = 0, seed = 12345;
    double ccomp = 0.5;                         // broken-bond compression stiffness factor
    // creep (viscoelastic LPS)
    bool creep = false;
    double tauRef = 3600.0, TtauRef = 1700.0, Qact = 380e3, lamv = 1.0;
    // thermal loading
    double qmax = 25e3, kth = 3.0, Tref = 293.15, gsurf = 0.0192;
    std::string history;
    double dqmax = 0;                           // max power increment per step [W/m] (0: qmax/steps)
    double dtmax = 0.5;                         // max time increment per step with creep [h]
    int steps = 40;
    // stack / contact
    std::string pin = "auto";                   // none | z | full | auto
    double pressure = 0, mu = 0.5, cn = 1.0, ctang = 1.0, contactRmin = 0;
    // solver
    int maxit = 6000, maxcycles = 20000;
    double kappa = 0.8, tol = 1e-3, sf = 1.0;
    // analysis
    double dstress = 0.5, cover = 0.35, dcut = 0.35, onsetFrac = 0.05;
    int minfrag = 30;
    // io
    std::string out = "pd_out", restartIn, restartOut;
    int restartBody = -1, every = 5;
    bool elastic = false, novtk = false, nostate = false, verbose = false, dryrun = false;
    double gbes = 2.0;                          // assumed G bond-evals/s for --dryrun estimate
    int ckptEvery = 0;                          // checkpoint every n load steps (0 = off)
    bool resume = false;                        // continue from <out>/checkpoint.bin if present
    std::vector<std::pair<std::string, std::string>> labels;   // --label key=value (repeatable)
};

static void usage() {
    std::printf(
        "%s  -- peridynamic thermal cracking of fuel pellets / pellet stacks\n"
        "GEOMETRY  --R --H --dishR --dishD --chamfer [m]  --stack 0.5,1,0.5  --jitter f\n"
        "MESH      --nr N | --delta d [--mfac m]  (--delta fixes the physical horizon)\n"
        "MATERIAL  --E --nu --alpha --Gc  [--sc s | --sigmaf Pa] [--calib-delta]\n"
        "          --weibull m (<=0 off) --wcorr L (grain size, 0 = per bond) --seed s --ccomp f\n"
        "CREEP     --creep --tauRef s --TtauRef K --Qact J/mol --lamv f\n"
        "THERMAL   --qmax W/m --kth --Tref --gsurf K.m/W --history file(t_h q [p])\n"
        "          --steps n | --dqmax W/m  --dtmax h\n"
        "STACK     --pin none|z|full|auto --pressure Pa --mu f (<0 stick) --cn --ctang\n"
        "          --contact-rmin m --restart file --restart-body b --restart-out file\n"
        "SOLVER    --maxit --maxcycles --kappa --tol --sf --elastic\n"
        "ANALYSIS  --dstress --cover --dcut --minfrag --onsetFrac\n"
        "OUTPUT    --out dir --every n --novtk --nostate --verbose --dryrun [--gbes G]\n"
        "CLOUD     --ckpt-every n --resume   (SIGTERM -> exit 75 after the current ADR iteration)\n"
        "LABELS    --label key=value (repeatable; copied into manifest, summary and state headers)\n",
        PD_VERSION);
}

static std::vector<std::string> splitList(const std::string& s, char sep) {
    std::vector<std::string> r;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(sep, a);
        if (b == std::string::npos) b = s.size();
        if (b > a) r.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return r;
}

static bool parseArgs(int argc, char** argv, Params& p, std::string& cmdline) {
    std::map<std::string, double*> dbl = {
        {"R", &p.R}, {"H", &p.H}, {"dishR", &p.dishR}, {"dishD", &p.dishD}, {"chamfer", &p.chamfer},
        {"mfac", &p.mfac}, {"delta", &p.delta}, {"jitter", &p.jitter}, {"E", &p.E}, {"nu", &p.nu},
        {"alpha", &p.alpha}, {"Gc", &p.Gc}, {"sc", &p.sc}, {"sigmaf", &p.sigmaf}, {"weibull", &p.weibull},
        {"wcorr", &p.wcorr}, {"seed", &p.seed}, {"ccomp", &p.ccomp}, {"tauRef", &p.tauRef},
        {"TtauRef", &p.TtauRef}, {"Qact", &p.Qact}, {"lamv", &p.lamv}, {"qmax", &p.qmax}, {"kth", &p.kth},
        {"Tref", &p.Tref}, {"gsurf", &p.gsurf}, {"dqmax", &p.dqmax}, {"dtmax", &p.dtmax},
        {"pressure", &p.pressure}, {"mu", &p.mu}, {"cn", &p.cn}, {"ctang", &p.ctang},
        {"contact-rmin", &p.contactRmin}, {"kappa", &p.kappa}, {"tol", &p.tol}, {"sf", &p.sf},
        {"dstress", &p.dstress}, {"cover", &p.cover}, {"dcut", &p.dcut}, {"onsetFrac", &p.onsetFrac},
        {"gbes", &p.gbes}};
    std::map<std::string, int*> itg = {{"nr", &p.nr}, {"steps", &p.steps}, {"maxit", &p.maxit},
                                       {"maxcycles", &p.maxcycles}, {"every", &p.every},
                                       {"minfrag", &p.minfrag}, {"restart-body", &p.restartBody},
                                       {"ckpt-every", &p.ckptEvery}};
    std::map<std::string, std::string*> str = {{"stack", &p.stack}, {"history", &p.history}, {"out", &p.out},
                                               {"restart", &p.restartIn}, {"restart-out", &p.restartOut},
                                               {"pin", &p.pin}};
    std::map<std::string, bool*> flg = {{"elastic", &p.elastic}, {"novtk", &p.novtk}, {"nostate", &p.nostate},
                                        {"verbose", &p.verbose}, {"dryrun", &p.dryrun}, {"creep", &p.creep},
                                        {"calib-delta", &p.calibDelta}, {"resume", &p.resume}};
    cmdline.clear();
    for (int i = 0; i < argc; ++i) { cmdline += argv[i]; cmdline += (i + 1 < argc) ? " " : ""; }
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        if (a.rfind("--", 0) != 0) { std::fprintf(stderr, "bad argument: %s\n", a.c_str()); return false; }
        std::string key = a.substr(2), val;
        if (flg.count(key)) { *flg[key] = true; continue; }
        if (key == "label" || key.rfind("label=", 0) == 0) {
            std::string kv = key == "label" ? (i + 1 < argc ? argv[++i] : "") : key.substr(6);
            const auto e = kv.find('=');
            if (e == std::string::npos || e == 0) { std::fprintf(stderr, "--label needs key=value\n"); return false; }
            p.labels.emplace_back(kv.substr(0, e), kv.substr(e + 1));
            continue;
        }
        auto eq = key.find('=');
        if (eq != std::string::npos) { val = key.substr(eq + 1); key = key.substr(0, eq); }
        else if (i + 1 < argc) val = argv[++i];
        else { std::fprintf(stderr, "missing value for %s\n", a.c_str()); return false; }
        try {
            if (dbl.count(key)) *dbl[key] = std::stod(val);
            else if (itg.count(key)) *itg[key] = std::stoi(val);
            else if (str.count(key)) *str[key] = val;
            else { std::fprintf(stderr, "unknown option --%s\n", key.c_str()); return false; }
        } catch (...) { std::fprintf(stderr, "bad value for --%s: %s\n", key.c_str(), val.c_str()); return false; }
    }
    return true;
}

// ----------------------------------------------------------------------------
// KD-tree for the one-off neighbour search
// ----------------------------------------------------------------------------
class KDTree {
public:
    KDTree(const double* pts, int n, int leaf = 12) : P_(pts), leaf_(leaf), idx_(n) {
        std::iota(idx_.begin(), idx_.end(), 0);
        nodes_.reserve(4 * (n / leaf + 1));
        if (n > 0) build(0, n);
    }
    template <class CB>
    void radius(const double* q, double r2, CB&& cb) const {
        if (nodes_.empty()) return;
        int stack[256], sp = 0;
        stack[sp++] = 0;
        while (sp) {
            const Node& nd = nodes_[stack[--sp]];
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                const double v = q[d] < nd.lo[d] ? nd.lo[d] - q[d] : (q[d] > nd.hi[d] ? q[d] - nd.hi[d] : 0.0);
                d2 += v * v;
            }
            if (d2 > r2) continue;
            if (nd.left < 0) {
                for (int k = nd.b; k < nd.e; ++k) {
                    const int j = idx_[k];
                    const double dx = P_[3 * j] - q[0], dy = P_[3 * j + 1] - q[1], dz = P_[3 * j + 2] - q[2];
                    const double s = dx * dx + dy * dy + dz * dz;
                    if (s <= r2) cb(j, s);
                }
            } else {
                stack[sp++] = nd.left;
                stack[sp++] = nd.right;
            }
        }
    }
private:
    struct Node { double lo[3], hi[3]; int b, e, left, right; };
    const double* P_;
    int leaf_;
    std::vector<int> idx_;
    std::vector<Node> nodes_;
    int build(int b, int e) {
        const int id = (int)nodes_.size();
        nodes_.emplace_back();
        Node nd;
        for (int d = 0; d < 3; ++d) { nd.lo[d] = 1e300; nd.hi[d] = -1e300; }
        for (int k = b; k < e; ++k)
            for (int d = 0; d < 3; ++d) {
                const double c = P_[3 * idx_[k] + d];
                nd.lo[d] = std::min(nd.lo[d], c);
                nd.hi[d] = std::max(nd.hi[d], c);
            }
        nd.b = b; nd.e = e; nd.left = nd.right = -1;
        if (e - b > leaf_) {
            int dim = 0;
            for (int d = 1; d < 3; ++d)
                if (nd.hi[d] - nd.lo[d] > nd.hi[dim] - nd.lo[dim]) dim = d;
            const int mid = (b + e) / 2;
            std::nth_element(idx_.begin() + b, idx_.begin() + mid, idx_.begin() + e,
                             [&](int a, int c) { return P_[3 * a + dim] < P_[3 * c + dim]; });
            nd.left = build(b, mid);
            nd.right = build(mid, e);
        }
        nodes_[id] = nd;
        return id;
    }
};

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline double u01(uint64_t h) { return ((h >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

template <class T>
static std::unique_ptr<T[]> uninit(size_t n) { return std::unique_ptr<T[]>(new T[n]); }

static inline bool inv3(const double A[9], double Ai[9], double& det) {
    const double c00 = A[4] * A[8] - A[5] * A[7], c01 = A[5] * A[6] - A[3] * A[8], c02 = A[3] * A[7] - A[4] * A[6];
    det = A[0] * c00 + A[1] * c01 + A[2] * c02;
    if (std::fabs(det) < 1e-300) return false;
    const double id = 1.0 / det;
    Ai[0] = c00 * id; Ai[1] = (A[2] * A[7] - A[1] * A[8]) * id; Ai[2] = (A[1] * A[5] - A[2] * A[4]) * id;
    Ai[3] = c01 * id; Ai[4] = (A[0] * A[8] - A[2] * A[6]) * id; Ai[5] = (A[2] * A[3] - A[0] * A[5]) * id;
    Ai[6] = c02 * id; Ai[7] = (A[1] * A[6] - A[0] * A[7]) * id; Ai[8] = (A[0] * A[4] - A[1] * A[3]) * id;
    return true;
}

static inline uint32_t bswap32(uint32_t u) {
    return (u >> 24) | ((u >> 8) & 0xFF00u) | ((u << 8) & 0xFF0000u) | (u << 24);
}
static void putBE(std::vector<uint32_t>& buf, float f) { uint32_t u; std::memcpy(&u, &f, 4); buf.push_back(bswap32(u)); }
static void putBE(std::vector<uint32_t>& buf, int32_t i) { buf.push_back(bswap32((uint32_t)i)); }

static double percentile(std::vector<double> v, double q) {
    if (v.empty()) return NAN;
    const size_t k = std::min(v.size() - 1, (size_t)std::floor(q * (v.size() - 1)));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

static std::string jesc(const std::string& in) {
    std::string o;
    for (char c : in) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}
static std::string labelsJSON(const std::vector<std::pair<std::string, std::string>>& L) {
    std::string s = "{";
    for (size_t k = 0; k < L.size(); ++k) s += (k ? "," : "") + std::string("\"") + jesc(L[k].first) + "\":\"" + jesc(L[k].second) + "\"";
    return s + "}";
}

struct Knot { double t, q, p; };
struct LoadStep { double t, dt, q, p; bool knot; };

struct Restart {
    int32_t N = 0, nr = 0, creep = 0;
    int64_t NB = 0;
    double hdr[10] = {0};   // R, Heff, dx, delta, dishR, dishD, chamfer, jitter, time_h, seed
    std::vector<double> x0, u;
    std::vector<int32_t> fam;
    std::vector<uint8_t> brk;
    std::vector<float> sc, eb;
};

// ----------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------
struct Model {
    Params p;
    std::string cmdline;
    int N = 0, NBODY = 0, nxy = 0, nzTot = 0, mainBody = 0;
    int64_t NB = 0;
    double dx = 0, delta = 0, V = 0, K = 0, mu = 0, lamL = 0, sc0 = 0, scG = 0, Ccal = 0, cbb = 0, Anorm = 0;
    double Wbulk = 0, kn = 0, kt = 0;
    std::vector<double> hb, bH;
    std::vector<int> bStart, bCount, bZ0, bNz;

    // particles
    std::vector<double> x0, u, v, f, fo, uzero, bext;
    std::vector<int> gi, gj, gk;
    std::vector<uint8_t> body, fixedDof;
    std::vector<double> lamb, m, theta, A, Bd, Hc, B15, beta, T, Wtot, gw;
    bool rmX = true, rmY = true, rmZ = true;
    int nTop = 0;

    // bonds (CSR)
    std::vector<int64_t> off;
    std::unique_ptr<int[]> nbr;
    std::unique_ptr<float[]> xl, wv, sc, eb;
    std::unique_ptr<uint8_t[]> brk;
    std::unique_ptr<uint16_t[]> rev;

    // contact pairs (stacks)
    struct Pair { int i, j, iface; float r, th; double sc[2], st[2], pn, tn, gap; };
    std::vector<Pair> pairs;

    // state
    double tnow = 0, qnow = 0, pnow = 0, dT0 = 0, Ts = 0;
    std::vector<float> sig, eps, dmg;
    std::vector<int> fragId;
    Restart rs;
    bool haveRestart = false;
    FILE* cyclog = nullptr;
    int curStep = 0;

    // ------------------------------------------------------------------
    [[noreturn]] static void die(const char* msg) { std::fprintf(stderr, "ERROR: %s\n", msg); std::exit(2); }

    bool inside(double x, double y, double zl, double Hb) const {
        const double r = std::hypot(x, y);
        if (r > p.R) return false;
        if (p.chamfer > 0) {
            const double de = std::min(zl, Hb - zl);
            if (r > p.R - p.chamfer + de) return false;
        }
        if (p.dishR > 0 && p.dishD > 0 && r < p.dishR) {
            const double Rs = (p.dishR * p.dishR + p.dishD * p.dishD) / (2.0 * p.dishD);
            const double zt = Hb + Rs - p.dishD, zb = -Rs + p.dishD;
            if (r * r + (zl - zt) * (zl - zt) < Rs * Rs) return false;
            if (r * r + (zl - zb) * (zl - zb) < Rs * Rs) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------
    void init() {
        K = p.E / (3.0 * (1.0 - 2.0 * p.nu));
        mu = p.E / (2.0 * (1.0 + p.nu));
        lamL = K - 2.0 * mu / 3.0;
        Ccal = 3.0 * mu + std::pow(0.75, 4) * (K - 5.0 * mu / 3.0);
        for (auto& s : splitList(p.stack, ',')) hb.push_back(std::stod(s));
        NBODY = (int)hb.size();
        if (NBODY < 1 || NBODY > MAXB) die("--stack must list 1..8 body heights");
        if (p.jitter < 0 || p.jitter > 0.3) die("--jitter must be in [0, 0.3]");

        if (p.calibDelta) {
            if (p.sigmaf <= 0) die("--calib-delta needs --sigmaf");
            const double s = p.sigmaf / p.E;
            p.delta = p.Gc / (Ccal * s * s);
        }
        if (p.delta > 0) p.nr = std::max(4, (int)std::lround(p.R * p.mfac / p.delta));
        dx = p.R / p.nr;
        delta = p.delta > 0 ? p.delta : p.mfac * dx;
        if (delta < 1.8 * dx) die("horizon must be >= 1.8 dx (needed for lattice-neighbour masks)");
        V = dx * dx * dx;
        scG = std::sqrt(p.Gc / (Ccal * delta));
        sc0 = p.sc > 0 ? p.sc : ((p.sigmaf > 0 && !p.calibDelta) ? p.sigmaf / p.E : scG);
        cbb = 18.0 * K / (M_PI * std::pow(delta, 4));
        Anorm = 5.0 / (M_PI * std::pow(delta, 5));      // crack area per (sum x V V) of cut pairs
        kn = p.cn * p.E / dx;
        kt = p.ctang * kn;
        if (p.pin == "auto") p.pin = (NBODY > 1) ? "z" : "none";
        if (p.pin != "none" && p.pin != "z" && p.pin != "full") die("--pin must be none|z|full|auto");
        if (p.pressure > 0 && p.pin == "none") die("--pressure needs a pinned bottom (--pin z|full)");
        if (p.restartBody < 0) p.restartBody = (NBODY >= 3) ? 1 : 0;

        // ---- particles, body by body, on one global lattice ----
        nxy = 2 * p.nr;
        int zoff = 0;
        double bestH = -1;
        for (int b = 0; b < NBODY; ++b) {
            const int nzb = std::max(2, (int)std::lround(hb[b] * p.H / dx));
            const double Hb = nzb * dx;
            bStart.push_back(N); bZ0.push_back(zoff); bNz.push_back(nzb); bH.push_back(Hb);
            if (Hb > bestH) { bestH = Hb; mainBody = b; }
            const uint64_t sj = splitmix64((uint64_t)p.seed ^ 0xA5A5A5A5ULL);
            for (int k = 0; k < nzb; ++k)
                for (int j = 0; j < nxy; ++j)
                    for (int i = 0; i < nxy; ++i) {
                        const double x = -p.R + (i + 0.5) * dx, y = -p.R + (j + 0.5) * dx, zl = (k + 0.5) * dx;
                        if (!inside(x, y, zl, Hb)) continue;
                        double px = x, py = y, pz = (zoff + k + 0.5) * dx;
                        if (p.jitter > 0) {   // keyed on body-LOCAL lattice index -> reproducible per body
                            const uint64_t h = splitmix64(sj ^ ((uint64_t)k << 40) ^ ((uint64_t)j << 20) ^ (uint64_t)i);
                            px += p.jitter * dx * (u01(splitmix64(h + 1)) - 0.5);
                            py += p.jitter * dx * (u01(splitmix64(h + 2)) - 0.5);
                            pz += p.jitter * dx * (u01(splitmix64(h + 3)) - 0.5);
                        }
                        x0.push_back(px); x0.push_back(py); x0.push_back(pz);
                        gi.push_back(i); gj.push_back(j); gk.push_back(zoff + k);
                        body.push_back((uint8_t)b);
                        ++N;
                    }
            bCount.push_back(N - bStart[b]);
            zoff += nzb;
        }
        nzTot = zoff;
        if (!p.restartIn.empty()) loadRestartPhase1();

        u.assign(3 * (size_t)N, 0.0); v.assign(3 * (size_t)N, 0.0); f.assign(3 * (size_t)N, 0.0);
        fo.assign(3 * (size_t)N, 0.0); uzero.assign(3 * (size_t)N, 0.0); bext.assign(3 * (size_t)N, 0.0);
        fixedDof.assign(3 * (size_t)N, 0);
        lamb.assign(N, 0.0); m.assign(N, 0.0); theta.assign(N, 0.0); A.assign(N, 0.0); Bd.assign(N, 0.0);
        Hc.assign(N, 0.0); B15.assign(N, 0.0); beta.assign(N, 0.0); T.assign(N, p.Tref); Wtot.assign(N, 0.0);
        gw.assign(N, 1.0);
        fragId.assign(N, -1);
    }

    // ------------------------------------------------------------------
    // restart I/O (single-pellet state, restored into one body of a stack)
    // ------------------------------------------------------------------
    void writeRestart(const std::string& fn) const {
        if (NBODY != 1) { std::fprintf(stderr, "WARNING: --restart-out only supported for single pellets\n"); return; }
        FILE* fp = std::fopen(fn.c_str(), "wb");
        if (!fp) { std::fprintf(stderr, "cannot write %s\n", fn.c_str()); return; }
        const char magic[8] = {'P', 'D', 'R', 'S', 'T', '0', '2', 0};
        std::fwrite(magic, 1, 8, fp);
        const int32_t h32[3] = {N, p.nr, p.creep ? 1 : 0};
        std::fwrite(h32, 4, 3, fp);
        std::fwrite(&NB, 8, 1, fp);
        const double hdr[10] = {p.R, bH[0], dx, delta, p.dishR, p.dishD, p.chamfer, p.jitter, tnow, p.seed};
        std::fwrite(hdr, 8, 10, fp);
        std::fwrite(x0.data(), 8, 3 * (size_t)N, fp);
        std::fwrite(u.data(), 8, 3 * (size_t)N, fp);
        std::vector<int32_t> fam(N);
        for (int i = 0; i < N; ++i) fam[i] = (int32_t)(off[i + 1] - off[i]);
        std::fwrite(fam.data(), 4, N, fp);
        std::fwrite(brk.get(), 1, NB, fp);
        std::fwrite(sc.get(), 4, NB, fp);
        if (p.creep) std::fwrite(eb.get(), 4, NB, fp);
        std::fclose(fp);
    }

    void loadRestartPhase1() {
        FILE* fp = std::fopen(p.restartIn.c_str(), "rb");
        if (!fp) die("cannot open restart file");
        char magic[8];
        bool ok = std::fread(magic, 1, 8, fp) == 8 && std::memcmp(magic, "PDRST02", 7) == 0;
        int32_t h32[3];
        ok = ok && std::fread(h32, 4, 3, fp) == 3;
        rs.N = h32[0]; rs.nr = h32[1]; rs.creep = h32[2];
        ok = ok && std::fread(&rs.NB, 8, 1, fp) == 1 && std::fread(rs.hdr, 8, 10, fp) == 10;
        if (!ok) die("bad restart header");
        rs.x0.resize(3 * (size_t)rs.N); rs.u.resize(3 * (size_t)rs.N); rs.fam.resize(rs.N);
        rs.brk.resize(rs.NB); rs.sc.resize(rs.NB);
        ok = std::fread(rs.x0.data(), 8, rs.x0.size(), fp) == rs.x0.size() &&
             std::fread(rs.u.data(), 8, rs.u.size(), fp) == rs.u.size() &&
             std::fread(rs.fam.data(), 4, rs.N, fp) == (size_t)rs.N &&
             std::fread(rs.brk.data(), 1, rs.NB, fp) == (size_t)rs.NB &&
             std::fread(rs.sc.data(), 4, rs.NB, fp) == (size_t)rs.NB;
        if (ok && rs.creep) { rs.eb.resize(rs.NB); ok = std::fread(rs.eb.data(), 4, rs.NB, fp) == (size_t)rs.NB; }
        std::fclose(fp);
        if (!ok) die("truncated restart file");
        const int b = p.restartBody;
        if (b < 0 || b >= NBODY) die("--restart-body out of range");
        if (rs.N != bCount[b]) die("restart particle count differs from target body (geometry/mesh mismatch)");
        if (std::fabs(rs.hdr[2] - dx) > 1e-12 * dx || std::fabs(rs.hdr[3] - delta) > 1e-9 * delta)
            die("restart dx/delta differ from this run");
        const double zshift = bZ0[b] * dx;
        double dev = 0;
        for (int i = 0; i < rs.N; ++i) {
            const size_t gI = (size_t)(bStart[b] + i);
            dev = std::max(dev, std::fabs(x0[3 * gI] - rs.x0[3 * (size_t)i]) + std::fabs(x0[3 * gI + 1] - rs.x0[3 * (size_t)i + 1]));
            for (int d = 0; d < 3; ++d) x0[3 * gI + d] = rs.x0[3 * (size_t)i + d] + (d == 2 ? zshift : 0.0);
        }
        if (dev > 0.4 * dx) die("restart lattice layout differs from target body");
        haveRestart = true;
    }

    void loadRestartPhase2() {
        const int b = p.restartBody;
        const int s = bStart[b];
        int64_t qq = 0;
        for (int i = 0; i < rs.N; ++i) {
            const int gI = s + i;
            if (off[gI + 1] - off[gI] != rs.fam[i]) die("restart bond families differ (geometry mismatch)");
            for (int64_t q = off[gI]; q < off[gI + 1]; ++q, ++qq) {
                brk[q] = rs.brk[qq];
                sc[q] = rs.sc[qq];
                if (p.creep) eb[q] = rs.creep ? rs.eb[qq] : 0.f;
            }
            for (int d = 0; d < 3; ++d) u[3 * (size_t)gI + d] = rs.u[3 * (size_t)i + d];
        }
        // the source run was a free body: remove its arbitrary rigid translation so the restored
        // pellet sits on the lattice position of the target body (no spurious interface overlap)
        double mean[3] = {0, 0, 0};
        for (int i = 0; i < rs.N; ++i)
            for (int d = 0; d < 3; ++d) mean[d] += rs.u[3 * (size_t)i + d];
        for (int i = 0; i < rs.N; ++i)
            for (int d = 0; d < 3; ++d) u[3 * (size_t)(s + i) + d] -= mean[d] / rs.N;
        // free restart memory
        rs.brk.clear(); rs.brk.shrink_to_fit(); rs.sc.clear(); rs.sc.shrink_to_fit();
        rs.eb.clear(); rs.eb.shrink_to_fit();
    }

    // ------------------------------------------------------------------
    // KD-tree neighbour search -> CSR bonds (within a body only)
    // ------------------------------------------------------------------
    double tKD = 0, tBond = 0;

    void countBonds(KDTree& tree, std::vector<int>& cnt) {
        const double rs2 = (delta + 0.5 * dx) * (delta + 0.5 * dx);
#pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < N; ++i) {
            int c = 0;
            const uint8_t bi = body[i];
            tree.radius(&x0[3 * (size_t)i], rs2, [&](int j, double) { if (j != i && body[j] == bi) ++c; });
            cnt[i] = c;
        }
    }

    void buildBonds() {
        const double t0 = omp_get_wtime();
        KDTree tree(x0.data(), N);
        const double t1 = omp_get_wtime();
        std::vector<int> cnt(N);
        countBonds(tree, cnt);
        off.assign(N + 1, 0);
        for (int i = 0; i < N; ++i) {
            if (cnt[i] > 65535) die("family > 65535 bonds; reduce mfac");
            off[i + 1] = off[i] + cnt[i];
        }
        NB = off[N];
        nbr = uninit<int>(NB); xl = uninit<float>(NB); wv = uninit<float>(NB);
        sc = uninit<float>(NB); brk = uninit<uint8_t>(NB);
        if (p.creep) { eb = uninit<float>(NB); rev = uninit<uint16_t>(NB); }

        // grain field for spatially correlated Weibull scatter (mesh independent)
        const double wb = p.weibull;
        const double gnorm = wb > 0 ? std::tgamma(1.0 + 1.0 / wb) : 1.0;
        const uint64_t sd = splitmix64((uint64_t)p.seed);
        auto weib = [&](double U) { return std::pow(-std::log(1.0 - U), 1.0 / wb) / gnorm; };
        if (wb > 0 && p.wcorr > 0) {
            double sh[3];
            for (int d = 0; d < 3; ++d) sh[d] = u01(splitmix64(sd + 101 + d)) * p.wcorr;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                const double zl = x0[3 * (size_t)i + 2] - bZ0[body[i]] * dx;   // body-local -> restart-stable
                const int64_t cx = (int64_t)std::floor((x0[3 * (size_t)i] + p.R + sh[0]) / p.wcorr);
                const int64_t cy = (int64_t)std::floor((x0[3 * (size_t)i + 1] + p.R + sh[1]) / p.wcorr);
                const int64_t cz = (int64_t)std::floor((zl + sh[2]) / p.wcorr);
                uint64_t h = splitmix64(sd ^ (uint64_t)(cx + (1 << 20)));
                h = splitmix64(h ^ (uint64_t)(cy + (1 << 20)));
                h = splitmix64(h ^ (uint64_t)(cz + (1 << 20)) ^ ((uint64_t)body[i] << 48));
                gw[i] = weib(u01(h));
            }
        }

#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            int64_t q = off[i];
            const uint8_t bi = body[i];
            const double rs2 = (delta + 0.5 * dx) * (delta + 0.5 * dx);
            tree.radius(&x0[3 * (size_t)i], rs2, [&](int j, double) { if (j != i && body[j] == bi) nbr[q++] = j; });
            std::sort(&nbr[off[i]], &nbr[off[i + 1]]);
            double mi = 0.0, wt = 0.0;
            for (int64_t q2 = off[i]; q2 < off[i + 1]; ++q2) {
                const int j = nbr[q2];
                const double X0 = x0[3 * (size_t)j] - x0[3 * (size_t)i], X1 = x0[3 * (size_t)j + 1] - x0[3 * (size_t)i + 1],
                             X2 = x0[3 * (size_t)j + 2] - x0[3 * (size_t)i + 2];
                const double L = std::sqrt(X0 * X0 + X1 * X1 + X2 * X2);
                double betaV = 1.0;
                if (L > delta - 0.5 * dx) betaV = std::clamp((delta + 0.5 * dx - L) / dx, 0.0, 1.0);
                xl[q2] = (float)L;
                wv[q2] = (float)(betaV * V);
                brk[q2] = 0;
                double w = 1.0;
                if (wb > 0) {
                    if (p.wcorr > 0) w = std::min(gw[i], gw[j]);   // weakest grain across a boundary
                    else {
                        const uint64_t lo = (uint64_t)std::min(i, j), hi = (uint64_t)std::max(i, j);
                        w = weib(u01(splitmix64((lo << 32 | hi) ^ sd)));
                    }
                }
                sc[q2] = (float)(sc0 * w);
                if (p.creep) eb[q2] = 0.f;
                const double ww = wv[q2], Lf = xl[q2];
                mi += ww * Lf * Lf;
                wt += ww;
            }
            m[i] = mi; Wtot[i] = wt; B15[i] = 15.0 * mu / mi;
        }
        for (int i = 0; i < N; ++i) Wbulk = std::max(Wbulk, Wtot[i]);

        if (p.creep) {   // reverse-bond index (local position of i in j's sorted family)
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i)
                for (int64_t q = off[i]; q < off[i + 1]; ++q) {
                    const int j = nbr[q];
                    const int* a = &nbr[off[j]];
                    const int* e = &nbr[off[j + 1]];
                    const int* it = std::lower_bound(a, e, i);
                    rev[q] = (uint16_t)(it - a);
                }
        }

        buildContactAndBCs();

        // ADR fictitious mass (Gershgorin-type LPS estimate + contact)
        std::vector<double> Wx(N);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            double s = 0;
            for (int64_t q = off[i]; q < off[i + 1]; ++q) s += (double)wv[q] * xl[q];
            Wx[i] = s;
        }
        const double cdil = 9.0 * std::fabs(3.0 * K - 5.0 * mu);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            double st = 0;
            for (int64_t q = off[i]; q < off[i + 1]; ++q) {
                const int j = nbr[q];
                st += (double)wv[q] * (B15[i] + B15[j] + cdil * xl[q] * (Wx[i] / (m[i] * m[i]) + Wx[j] / (m[j] * m[j])));
            }
            lamb[i] = 0.25 * p.sf * st;
        }
        for (auto& c : pairs) {
            const double add = 0.25 * p.sf * 2.0 * (kn + kt) / dx;
            lamb[c.i] += add; lamb[c.j] += add;
        }
        if (haveRestart) loadRestartPhase2();
        tKD = t1 - t0;
        tBond = omp_get_wtime() - t1;
    }

    // contact pairs between consecutive bodies, pins, top-face particles
    void buildContactAndBCs() {
        const int ncol = nxy * nxy;
        std::vector<int> top((size_t)NBODY * ncol, -1), bot((size_t)NBODY * ncol, -1);
        for (int i = 0; i < N; ++i) {
            const size_t c = (size_t)body[i] * ncol + (size_t)gj[i] * nxy + gi[i];
            if (top[c] < 0 || gk[i] > gk[top[c]]) top[c] = i;
            if (bot[c] < 0 || gk[i] < gk[bot[c]]) bot[c] = i;
        }
        std::vector<int> owner(N, -1);
        for (int b = 0; b + 1 < NBODY; ++b)
            for (int c = 0; c < ncol; ++c) {
                const int i = top[(size_t)b * ncol + c], j = bot[(size_t)(b + 1) * ncol + c];
                if (i < 0 || j < 0) continue;
                const double xr = -p.R + (c % nxy + 0.5) * dx, yr = -p.R + (c / nxy + 0.5) * dx;
                const double r = std::hypot(xr, yr);
                if (r < p.contactRmin) continue;
                if (owner[i] >= 0 || owner[j] >= 0) continue;   // body too thin in this column
                Pair pr{};
                pr.i = i; pr.j = j; pr.iface = b; pr.r = (float)r; pr.th = (float)std::atan2(yr, xr);
                owner[i] = owner[j] = (int)pairs.size();
                pairs.push_back(pr);
            }
        // pins on the lowest layer of body 0
        if (p.pin != "none")
            for (int i = 0; i < N; ++i)
                if (body[i] == 0 && gk[i] == bZ0[0]) {
                    fixedDof[3 * (size_t)i + 2] = 1;
                    if (p.pin == "full") fixedDof[3 * (size_t)i] = fixedDof[3 * (size_t)i + 1] = 1;
                }
        rmZ = (p.pin == "none");
        rmX = rmY = (p.pin != "full");
        // top-face particles of the last body (pressure)
        const int lb = NBODY - 1, ktop = bZ0[lb] + bNz[lb] - 1;
        nTop = 0;
        for (int i = 0; i < N; ++i) if (body[i] == lb && gk[i] == ktop) ++nTop;
    }

    // ------------------------------------------------------------------
    // Load schedule from a history file or the built-in ramp
    // ------------------------------------------------------------------
    std::vector<Knot> knots;
    std::vector<LoadStep> sched;

    void buildSchedule() {
        if (!p.history.empty()) {
            FILE* fp = std::fopen(p.history.c_str(), "r");
            if (!fp) die("cannot open --history file");
            char line[1024];
            while (std::fgets(line, sizeof line, fp)) {
                if (char* c = std::strchr(line, '#')) *c = 0;
                for (char* c = line; *c; ++c) if (*c == ',' || *c == ';' || *c == '\t') *c = ' ';
                double a, b, c;
                const int n = std::sscanf(line, "%lf %lf %lf", &a, &b, &c);
                if (n >= 2) knots.push_back({a, b, n >= 3 ? c : p.pressure});
            }
            std::fclose(fp);
            if (knots.size() < 2) die("history needs >= 2 rows: time_h q_Wpm [p_Pa]");
            for (size_t k = 1; k < knots.size(); ++k) if (knots[k].t < knots[k - 1].t) die("history time must not decrease");
        } else {
            knots.push_back({0.0, 0.0, p.pressure});
            knots.push_back({1.0, p.qmax, p.pressure});
        }
        double qpk = 0;
        for (auto& k : knots) qpk = std::max(qpk, k.q);
        const double dq = p.dqmax > 0 ? p.dqmax : std::max(1.0, qpk / std::max(1, p.steps));
        if (knots[0].p > 0 || knots[0].q > 0) sched.push_back({knots[0].t, 0.0, knots[0].q, knots[0].p, true});  // preload
        for (size_t k = 1; k < knots.size(); ++k) {
            const Knot& a = knots[k - 1];
            const Knot& b = knots[k];
            int n = std::max(1, (int)std::ceil(std::fabs(b.q - a.q) / dq - 1e-9));
            if (p.creep && b.t > a.t) n = std::max(n, (int)std::ceil((b.t - a.t) / p.dtmax - 1e-9));
            for (int s = 1; s <= n; ++s) {
                const double fr = (double)s / n;
                sched.push_back({a.t + fr * (b.t - a.t), (b.t - a.t) / n, a.q + fr * (b.q - a.q), a.p + fr * (b.p - a.p), s == n});
            }
        }
    }

    // ------------------------------------------------------------------
    // per-step fields: temperature, creep factors, pressure load
    // ------------------------------------------------------------------
    void prepareStep(const LoadStep& L) {
        qnow = L.q; pnow = L.p; tnow = L.t;
        dT0 = L.q / (4.0 * M_PI * p.kth);
        Ts = p.Tref + p.gsurf * L.q;
        const double R2 = p.R * p.R, dts = L.dt * 3600.0;
        const double TsL = Ts, dT0L = dT0;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const double r2 = x0[3 * (size_t)i] * x0[3 * (size_t)i] + x0[3 * (size_t)i + 1] * x0[3 * (size_t)i + 1];
            T[i] = TsL + dT0L * std::max(0.0, 1.0 - r2 / R2);
            double bt = 0.0;
            if (p.creep && dts > 0) {
                const double tau = p.tauRef * std::exp(p.Qact / RGAS * (1.0 / T[i] - 1.0 / p.TtauRef));
                bt = -std::expm1(-dts / tau);
            }
            beta[i] = bt;
            Bd[i] = B15[i] * (1.0 - p.lamv * bt);
            Hc[i] = p.creep ? B15[i] * p.lamv * (1.0 - bt) : 0.0;
        }
        std::fill(bext.begin(), bext.end(), 0.0);
        if (pnow > 0 && nTop > 0) {
            const int lb = NBODY - 1, ktop = bZ0[lb] + bNz[lb] - 1;
            const double fd = pnow * M_PI * p.R * p.R / (nTop * V);   // nominal pressure on full end area
            for (int i = 0; i < N; ++i) if (body[i] == lb && gk[i] == ktop) bext[3 * (size_t)i + 2] = -fd;
        }
    }

    // ------------------------------------------------------------------
    // Internal + contact + external force density
    // ------------------------------------------------------------------
    void forces(const double* __restrict U, double* __restrict F) {
        const double alpha = p.alpha, Tref = p.Tref, threeK = 3.0 * K, cc = p.ccomp * cbb;
        const bool creep = p.creep;
        const double* __restrict X = x0.data();
        const double* __restrict Tp = T.data();
        const int* __restrict NBp = nbr.get();
        const float* __restrict XL = xl.get();
        const float* __restrict WV = wv.get();
        const uint8_t* __restrict BR = brk.get();
        const float* __restrict EB = creep ? eb.get() : nullptr;
        const uint16_t* __restrict RV = creep ? rev.get() : nullptr;
        const int64_t* __restrict OF = off.data();
        const double* __restrict M = m.data();
        double* __restrict TH = theta.data();
        double* __restrict Ap = A.data();
        const double* __restrict BD = Bd.data();
        const double* __restrict HC = Hc.data();
        const double* __restrict BX = bext.data();
        const int npairs = (int)pairs.size();
        const double kn_ = kn, kt_ = kt, muf = p.mu, dxl = dx;

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                const double yi0 = X[3 * i] + U[3 * i], yi1 = X[3 * i + 1] + U[3 * i + 1], yi2 = X[3 * i + 2] + U[3 * i + 2];
                const double Ti = Tp[i];
                double th = 0.0;
                for (int64_t q = OF[i]; q < OF[i + 1]; ++q) {
                    if (BR[q]) continue;
                    const int j = NBp[q];
                    const double d0 = X[3 * j] + U[3 * j] - yi0, d1 = X[3 * j + 1] + U[3 * j + 1] - yi1,
                                 d2 = X[3 * j + 2] + U[3 * j + 2] - yi2;
                    const double y = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
                    const double x = XL[q];
                    th += (double)WV[q] * x * (y - x * (1.0 + alpha * (0.5 * (Ti + Tp[j]) - Tref)));
                }
                const double t = 3.0 * th / M[i];
                TH[i] = t;
                Ap[i] = threeK * t / M[i] - BD[i] * t / 3.0;   // t_ij = A_i x + Bd_i e - Hc_i e_b
            }
#pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                const double yi0 = X[3 * i] + U[3 * i], yi1 = X[3 * i + 1] + U[3 * i + 1], yi2 = X[3 * i + 2] + U[3 * i + 2];
                const double Ti = Tp[i], Ai = Ap[i], Bi = BD[i], Hi = HC[i];
                double f0 = 0, f1 = 0, f2 = 0;
                for (int64_t q = OF[i]; q < OF[i + 1]; ++q) {
                    const int j = NBp[q];
                    const double d0 = X[3 * j] + U[3 * j] - yi0, d1 = X[3 * j + 1] + U[3 * j + 1] - yi1,
                                 d2 = X[3 * j + 2] + U[3 * j + 2] - yi2;
                    const double y = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
                    const double x = XL[q];
                    const double e = y - x * (1.0 + alpha * (0.5 * (Ti + Tp[j]) - Tref));
                    double s;
                    if (BR[q]) {
                        if (e >= 0.0) continue;
                        s = cc * (e / x) * (double)WV[q] / y;          // crack faces in contact
                    } else {
                        double tsum = (Ai + Ap[j]) * x + (Bi + BD[j]) * e;
                        if (creep) tsum -= Hi * EB[q] + HC[j] * EB[OF[j] + RV[q]];
                        s = tsum * (double)WV[q] / y;
                    }
                    f0 += s * d0; f1 += s * d1; f2 += s * d2;
                }
                F[3 * i] = f0 + BX[3 * i]; F[3 * i + 1] = f1 + BX[3 * i + 1]; F[3 * i + 2] = f2 + BX[3 * i + 2];
            }
            // inter-body contact: each particle belongs to at most one pair -> race free
#pragma omp for schedule(static)
            for (int k = 0; k < npairs; ++k) {
                Pair& c = pairs[k];
                const int i = c.i, j = c.j;
                const double dz = (X[3 * j + 2] + U[3 * j + 2]) - (X[3 * i + 2] + U[3 * i + 2]);
                const double gap = dz - dxl * (1.0 + alpha * (0.5 * (Tp[i] + Tp[j]) - Tref));
                const double r0 = U[3 * j] - U[3 * i], r1 = U[3 * j + 1] - U[3 * i + 1];
                double pn = 0, t0 = 0, t1 = 0;
                if (gap < 0) {
                    pn = kn_ * (-gap);
                    t0 = kt_ * (r0 - c.sc[0]);
                    t1 = kt_ * (r1 - c.sc[1]);
                    const double tm = std::sqrt(t0 * t0 + t1 * t1);
                    if (muf >= 0 && tm > muf * pn) {             // Coulomb slip: return mapping
                        const double sfac = muf * pn / tm;
                        t0 *= sfac; t1 *= sfac;
                    }
                    c.st[0] = r0 - t0 / kt_;
                    c.st[1] = r1 - t1 / kt_;
                } else {
                    c.st[0] = r0; c.st[1] = r1;                    // open: no traction, slip follows
                }
                c.pn = pn; c.tn = std::sqrt(t0 * t0 + t1 * t1); c.gap = gap;
                F[3 * j] -= t0 / dxl; F[3 * j + 1] -= t1 / dxl; F[3 * j + 2] += pn / dxl;
                F[3 * i] += t0 / dxl; F[3 * i + 1] += t1 / dxl; F[3 * i + 2] -= pn / dxl;
            }
        }
    }

    // max over particles of |f_i| on free DOFs (reactions excluded)
    double residInf(const std::vector<double>& w) const {
        double s = 0;
#pragma omp parallel for reduction(max : s) schedule(static)
        for (int i = 0; i < N; ++i) {
            double a = 0;
            for (int d = 0; d < 3; ++d) if (!fixedDof[3 * (size_t)i + d]) a += w[3 * (size_t)i + d] * w[3 * (size_t)i + d];
            s = std::max(s, a);
        }
        return std::sqrt(s);
    }

    void commitSlip() { for (auto& c : pairs) { c.sc[0] = c.st[0]; c.sc[1] = c.st[1]; } }

    // ------------------------------------------------------------------
    // Adaptive dynamic relaxation
    // ------------------------------------------------------------------
    struct StepInfo { int iters; double res; };

    StepInfo relax(double scale) {
        StepInfo info{0, 1.0};
        const int64_t n3 = 3 * (int64_t)N;
        forces(u.data(), f.data());
        info.res = residInf(f) / scale;
        if (info.res < p.tol) { commitSlip(); return info; }
#pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < n3; ++k) {
            v[k] = fixedDof[k] ? 0.0 : 0.5 * f[k] / lamb[k / 3];
            u[k] += v[k];
            fo[k] = f[k];
        }
        int nfree = 0;
        for (int i = 0; i < N; ++i) nfree += !fixedDof[3 * (size_t)i + 2] || !fixedDof[3 * (size_t)i];
        for (int it = 1; it <= p.maxit; ++it) {
            forces(u.data(), f.data());
            double num = 0, den = 0;
#pragma omp parallel for reduction(+ : num, den) schedule(static)
            for (int64_t k = 0; k < n3; ++k) {
                if (fixedDof[k]) continue;
                if (v[k] != 0.0) num -= u[k] * u[k] * (f[k] - fo[k]) / (lamb[k / 3] * v[k]);
                den += u[k] * u[k];
            }
            double c = (den > 0 && num > 0) ? 2.0 * std::sqrt(num / den) : 0.0;
            if (c > 1.999) c = 1.999;
            double vm0 = 0, vm1 = 0, vm2 = 0, fn = 0;
#pragma omp parallel for reduction(+ : vm0, vm1, vm2) reduction(max : fn) schedule(static)
            for (int i = 0; i < N; ++i) {
                const double li = lamb[i];
                double a = 0;
                for (int d = 0; d < 3; ++d) {
                    const size_t k = 3 * (size_t)i + d;
                    if (fixedDof[k]) { v[k] = 0.0; fo[k] = f[k]; continue; }
                    v[k] = ((2.0 - c) * v[k] + 2.0 * f[k] / li) / (2.0 + c);
                    fo[k] = f[k];
                    a += f[k] * f[k];
                }
                fn = std::max(fn, a);
                vm0 += v[3 * (size_t)i]; vm1 += v[3 * (size_t)i + 1]; vm2 += v[3 * (size_t)i + 2];
            }
            const double inv = 1.0 / std::max(1, N);
            vm0 = rmX ? vm0 * inv : 0.0; vm1 = rmY ? vm1 * inv : 0.0; vm2 = rmZ ? vm2 * inv : 0.0;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                const size_t k = 3 * (size_t)i;
                if (!fixedDof[k]) v[k] -= vm0;
                if (!fixedDof[k + 1]) v[k + 1] -= vm1;
                if (!fixedDof[k + 2]) v[k + 2] -= vm2;
                u[k] += v[k]; u[k + 1] += v[k + 1]; u[k + 2] += v[k + 2];
            }
            info.iters = it;
            info.res = std::sqrt(fn) / scale;
            if (it > 3 && info.res < p.tol) break;
            if (gStop) break;
        }
        (void)nfree;
        forces(u.data(), f.data());      // sync theta / contact trial state with final u
        commitSlip();
        return info;
    }

    // stress-carrying stretch ratio of an intact bond (symmetric in i, j)
    inline double bondRatio(int i, int64_t q) const {
        const int j = nbr[q];
        const double d0 = x0[3 * (size_t)j] + u[3 * (size_t)j] - x0[3 * (size_t)i] - u[3 * (size_t)i];
        const double d1 = x0[3 * (size_t)j + 1] + u[3 * (size_t)j + 1] - x0[3 * (size_t)i + 1] - u[3 * (size_t)i + 1];
        const double d2 = x0[3 * (size_t)j + 2] + u[3 * (size_t)j + 2] - x0[3 * (size_t)i + 2] - u[3 * (size_t)i + 2];
        const double y = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
        const double x = xl[q];
        double e = y - x * (1.0 + p.alpha * (0.5 * (T[i] + T[j]) - p.Tref));
        if (p.creep) e -= p.lamv * 0.5 * ((double)eb[q] + (double)eb[off[j] + rev[q]]);
        return e / ((double)sc[q] * x);
    }

    int64_t breakCritical(double& rmaxOut) {
        double rmax = 0;
#pragma omp parallel for reduction(max : rmax) schedule(static)
        for (int i = 0; i < N; ++i)
            for (int64_t q = off[i]; q < off[i + 1]; ++q)
                if (!brk[q]) rmax = std::max(rmax, bondRatio(i, q));
        rmaxOut = rmax;
        if (rmax <= 1.0) return 0;
        const double thr = std::max(1.0, p.kappa * rmax);
        int64_t nb = 0;
#pragma omp parallel for reduction(+ : nb) schedule(static)
        for (int i = 0; i < N; ++i)
            for (int64_t q = off[i]; q < off[i + 1]; ++q)
                if (!brk[q] && bondRatio(i, q) >= thr) { brk[q] = 1; ++nb; }
        return nb;
    }

    struct LoadInfo { int iters, cycles; double res; int64_t broken; double scale; };

    LoadInfo solveLoadStep() {
        forces(uzero.data(), f.data());
        double bmax = 0;
        for (int64_t k = 0; k < 3 * (int64_t)N; ++k) bmax = std::max(bmax, std::fabs(bext[k]));
        const double scale = std::max({residInf(f), bmax, 1e-300});
        auto r = relax(scale);
        LoadInfo L{r.iters, 0, r.res, 0, scale};
        if (p.elastic) return L;
        while (L.cycles < p.maxcycles && !gStop) {
            double rmax;
            const int64_t nb = breakCritical(rmax);
            if (nb == 0) break;
            L.broken += nb;
            ++L.cycles;
            r = relax(scale);
            L.iters += r.iters;
            L.res = r.res;
            if (cyclog) std::fprintf(cyclog, "%d,%d,%.5f,%lld,%d,%.4e\n", curStep, L.cycles, rmax, (long long)(nb / 2), r.iters, r.res);
            if (p.verbose)
                std::printf("        cycle %5d  max ratio %.3f  broke %6lld  relax iters %5d  resid %.2e\n", L.cycles, rmax,
                            (long long)(nb / 2), r.iters, r.res);
        }
        return L;
    }

    // creep: commit back-extensions at the converged state of the step
    void commitCreep() {
        if (!p.creep) return;
        forces(u.data(), f.data());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const double bi = beta[i];
            if (bi <= 0) continue;
            const double th = theta[i];
            for (int64_t q = off[i]; q < off[i + 1]; ++q) {
                if (brk[q]) continue;
                const int j = nbr[q];
                const double d0 = x0[3 * (size_t)j] + u[3 * (size_t)j] - x0[3 * (size_t)i] - u[3 * (size_t)i];
                const double d1 = x0[3 * (size_t)j + 1] + u[3 * (size_t)j + 1] - x0[3 * (size_t)i + 1] - u[3 * (size_t)i + 1];
                const double d2 = x0[3 * (size_t)j + 2] + u[3 * (size_t)j + 2] - x0[3 * (size_t)i + 2] - u[3 * (size_t)i + 2];
                const double x = xl[q];
                const double e = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2) - x * (1.0 + p.alpha * (0.5 * (T[i] + T[j]) - p.Tref));
                const double ed = e - th * x / 3.0;
                eb[q] = (float)(eb[q] + bi * (ed - eb[q]));
            }
        }
    }

    // ------------------------------------------------------------------
    // Post-processing
    // ------------------------------------------------------------------
    void computeStressStrain() {
        sig.assign(6 * (size_t)N, 0.f); eps.assign(6 * (size_t)N, 0.f); dmg.assign(N, 0.f);
        const float qnan = std::nanf("");
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            double Km[9] = {0}, Nm[9] = {0}, Nt[9] = {0}, wint = 0;
            int nint = 0;
            for (int64_t q = off[i]; q < off[i + 1]; ++q) {
                if (brk[q]) continue;
                const int j = nbr[q];
                const double w = wv[q];
                const double ath = p.alpha * (0.5 * (T[i] + T[j]) - p.Tref);
                double Xb[3], Yb[3];
                for (int d = 0; d < 3; ++d) {
                    Xb[d] = x0[3 * (size_t)j + d] - x0[3 * (size_t)i + d];
                    Yb[d] = Xb[d] + u[3 * (size_t)j + d] - u[3 * (size_t)i + d];
                }
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) {
                        Km[3 * r + c] += w * Xb[r] * Xb[c];
                        Nm[3 * r + c] += w * Yb[r] * Xb[c];
                        Nt[3 * r + c] += w * ath * Xb[r] * Xb[c];
                    }
                wint += w; ++nint;
            }
            dmg[i] = (float)(1.0 - wint / Wtot[i]);
            for (int c = 0; c < 6; ++c) { sig[6 * (size_t)i + c] = qnan; eps[6 * (size_t)i + c] = qnan; }
            if (dmg[i] > p.dstress || (dmg[i] > 0.05f && wint < p.cover * Wbulk)) continue;
            double Ki[9], det;
            const double trK = (Km[0] + Km[4] + Km[8]) / 3.0;
            if (nint < 8 || !inv3(Km, Ki, det) || det < 0.02 * trK * trK * trK) continue;
            double Fm[9], Ft[9];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    Fm[3 * r + c] = Nm[3 * r] * Ki[c] + Nm[3 * r + 1] * Ki[3 + c] + Nm[3 * r + 2] * Ki[6 + c];
                    Ft[3 * r + c] = Nt[3 * r] * Ki[c] + Nt[3 * r + 1] * Ki[3 + c] + Nt[3 * r + 2] * Ki[6 + c];
                }
            const double exx = Fm[0] - 1, eyy = Fm[4] - 1, ezz = Fm[8] - 1;
            const double eyz = 0.5 * (Fm[5] + Fm[7]), exz = 0.5 * (Fm[2] + Fm[6]), exy = 0.5 * (Fm[1] + Fm[3]);
            // elastic = total - thermal (thermal fitted over the same family, bond-mean T).
            // NOTE: with --creep this is the instantaneous elastic estimate; the relaxed
            // deviatoric part held in e_b is not subtracted here (see stress_* in history
            // for force-consistent percentiles).
            const double ex = exx - Ft[0], ey = eyy - Ft[4], ez = ezz - Ft[8];
            const double gyz = eyz - 0.5 * (Ft[5] + Ft[7]), gxz = exz - 0.5 * (Ft[2] + Ft[6]), gxy = exy - 0.5 * (Ft[1] + Ft[3]);
            const double tr = ex + ey + ez;
            float* s = &sig[6 * (size_t)i];
            float* e = &eps[6 * (size_t)i];
            s[0] = (float)(lamL * tr + 2 * mu * ex); s[1] = (float)(lamL * tr + 2 * mu * ey);
            s[2] = (float)(lamL * tr + 2 * mu * ez); s[3] = (float)(2 * mu * gyz);
            s[4] = (float)(2 * mu * gxz); s[5] = (float)(2 * mu * gxy);
            e[0] = (float)exx; e[1] = (float)eyy; e[2] = (float)ezz; e[3] = (float)eyz; e[4] = (float)exz; e[5] = (float)exy;
        }
    }

    void cyl(const float* t, int i, double& rr, double& tt, double& zz) const {
        const double x = x0[3 * (size_t)i], y = x0[3 * (size_t)i + 1];
        const double r = std::sqrt(x * x + y * y);
        double c = 1, s = 0;
        if (r > 1e-14) { c = x / r; s = y / r; }
        rr = c * c * t[0] + s * s * t[1] + 2 * c * s * t[5];
        tt = s * s * t[0] + c * c * t[1] - 2 * c * s * t[5];
        zz = t[2];
    }
    static double vonMises(const float* s) {
        const double a1 = s[0] - s[1], a2 = s[1] - s[2], a3 = s[2] - s[0];
        return std::sqrt(0.5 * (a1 * a1 + a2 * a2 + a3 * a3) + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
    }

    // crack area from cut bonds: a plane through an intact body cuts sum(x V V) = pi delta^5 / 5 per unit area
    void crackArea(double* Ab, int64_t* pairsB) const {
        double acc[MAXB] = {0};
        int64_t cntb[MAXB] = {0};
#pragma omp parallel for reduction(+ : acc[:MAXB], cntb[:MAXB]) schedule(static)
        for (int i = 0; i < N; ++i) {
            double a = 0; int64_t c = 0;
            for (int64_t q = off[i]; q < off[i + 1]; ++q)
                if (brk[q]) { a += (double)xl[q] * wv[q]; ++c; }
            acc[body[i]] += a; cntb[body[i]] += c;
        }
        for (int b = 0; b < NBODY; ++b) { Ab[b] = 0.5 * acc[b] * V * Anorm; pairsB[b] = cntb[b] / 2; }
    }

    int midLayer(int b) const { return bZ0[b] + bNz[b] / 2; }

    // number of radial cracks at the body mid-plane (angular runs of damage on the outer ring)
    int countRadialCracks(int b, std::vector<float>* prof = nullptr, int nbins = 96) const {
        std::vector<float> mx(nbins, 0.f);
        const int kl = midLayer(b);
        for (int i = bStart[b]; i < bStart[b] + bCount[b]; ++i) {
            if (gk[i] != kl) continue;
            const double r = std::hypot(x0[3 * (size_t)i], x0[3 * (size_t)i + 1]);
            if (r < 0.75 * p.R) continue;
            const double ang = std::atan2(x0[3 * (size_t)i + 1], x0[3 * (size_t)i]) + M_PI;
            const int k = std::min(nbins - 1, (int)(ang / (2 * M_PI) * nbins));
            mx[k] = std::max(mx[k], dmg[i]);
        }
        if (prof) *prof = mx;
        int runs = 0, on = 0;
        for (int k = 0; k < nbins; ++k) {
            const bool c = mx[k] > 0.4f, prev = mx[(k + nbins - 1) % nbins] > 0.4f;
            on += c;
            if (c && !prev) ++runs;
        }
        return (on == nbins) ? -1 : runs;
    }

    // fragments: connected components of intact bonds between particles with damage < dcut
    struct Frag { int body, id, n; double vol, deq, cx, cy, cz, rmin, rmax, zmin, zmax; };
    struct FragStats { int n = 0; double vtot = 0, vmean = 0, vmed = 0, deqMean = 0, deqVW = 0, largestFrac = 0, Leq = 0; };

    std::vector<Frag> fragments(FragStats* st, const double* Ab) {
        std::vector<int> par(N);
        std::iota(par.begin(), par.end(), 0);
        auto find = [&](int a) { while (par[a] != a) { par[a] = par[par[a]]; a = par[a]; } return a; };
        for (int i = 0; i < N; ++i) {
            if (dmg[i] >= p.dcut) continue;
            for (int64_t q = off[i]; q < off[i + 1]; ++q) {
                const int j = nbr[q];
                if (j <= i || brk[q] || dmg[j] >= p.dcut) continue;
                const int a = find(i), b = find(j);
                if (a != b) par[std::max(a, b)] = std::min(a, b);
            }
        }
        std::map<int, Frag> comp;
        for (int i = 0; i < N; ++i) {
            fragId[i] = -1;
            if (dmg[i] >= p.dcut) continue;
            const int r = find(i);
            Frag& F = comp[r];
            const double xx = x0[3 * (size_t)i], yy = x0[3 * (size_t)i + 1], zz = x0[3 * (size_t)i + 2], rr = std::hypot(xx, yy);
            if (F.n == 0) { F.body = body[i]; F.rmin = F.zmin = 1e300; F.rmax = F.zmax = -1e300; }
            F.n++; F.cx += xx; F.cy += yy; F.cz += zz;
            F.rmin = std::min(F.rmin, rr); F.rmax = std::max(F.rmax, rr);
            F.zmin = std::min(F.zmin, zz); F.zmax = std::max(F.zmax, zz);
        }
        std::vector<Frag> out;
        std::map<int, int> idOf;
        for (auto& kv : comp) {
            Frag F = kv.second;
            if (F.n < p.minfrag) continue;
            F.vol = F.n * V; F.deq = std::cbrt(6.0 * F.vol / M_PI);
            F.cx /= F.n; F.cy /= F.n; F.cz /= F.n;
            F.id = (int)out.size();
            idOf[kv.first] = F.id;
            out.push_back(F);
        }
        for (int i = 0; i < N; ++i)
            if (dmg[i] < p.dcut) { auto it = idOf.find(find(i)); if (it != idOf.end()) fragId[i] = it->second; }
        for (int b = 0; b < NBODY; ++b) {
            std::vector<double> vols;
            for (auto& F : out) if (F.body == b) vols.push_back(F.vol);
            FragStats S;
            S.n = (int)vols.size();
            if (S.n) {
                double s1 = 0, s2 = 0, sd = 0, vmax = 0;
                for (double vv : vols) { s1 += vv; s2 += vv * vv; sd += std::cbrt(6 * vv / M_PI); vmax = std::max(vmax, vv); }
                S.vtot = s1; S.vmean = s1 / S.n; S.vmed = percentile(vols, 0.5);
                S.deqMean = sd / S.n; S.deqVW = std::cbrt(6.0 * (s2 / s1) / M_PI); S.largestFrac = vmax / (bCount[b] * V);
            }
            S.Leq = Ab[b] > 0 ? 3.0 * bCount[b] * V / Ab[b] : NAN;
            st[b] = S;
        }
        return out;
    }

    // dimensional changes of body b: mean radial displacement of the outer ring at mid-plane and
    // at the end layers (hourglassing), axial elongation
    void dims(int b, double& urMid, double& urEnd, double& dH) const {
        const int kl = midLayer(b), k0 = bZ0[b], k1 = bZ0[b] + bNz[b] - 1;
        double sm = 0, se = 0, zt = 0, zb = 0; int nm = 0, ne = 0, nt = 0, nb = 0;
        for (int i = bStart[b]; i < bStart[b] + bCount[b]; ++i) {
            const double x = x0[3 * (size_t)i], y = x0[3 * (size_t)i + 1], r = std::hypot(x, y);
            if (r > p.R - 1.01 * dx && r > 1e-12) {
                const double ur = (u[3 * (size_t)i] * x + u[3 * (size_t)i + 1] * y) / r;
                if (gk[i] == kl) { sm += ur; ++nm; }
                if (gk[i] == k0 || gk[i] == k1) { se += ur; ++ne; }
            }
            if (gk[i] == k1) { zt += u[3 * (size_t)i + 2]; ++nt; }
            if (gk[i] == k0) { zb += u[3 * (size_t)i + 2]; ++nb; }
        }
        urMid = nm ? sm / nm : NAN; urEnd = ne ? se / ne : NAN;
        dH = (nt && nb) ? zt / nt - zb / nb : NAN;
    }

    // ------------------------------------------------------------------
    // Output writers
    // ------------------------------------------------------------------
    void writeVTK(const std::string& fn) const {
        FILE* fp = std::fopen(fn.c_str(), "wb");
        if (!fp) return;
        std::vector<uint32_t> buf;
        auto flush = [&]() { std::fwrite(buf.data(), 4, buf.size(), fp); buf.clear(); };
        std::fprintf(fp, "# vtk DataFile Version 3.0\nperidynamic fuel pellet\nBINARY\nDATASET POLYDATA\nPOINTS %d float\n", N);
        for (size_t k = 0; k < 3 * (size_t)N; ++k) putBE(buf, (float)x0[k]);
        flush();
        std::fprintf(fp, "\nVERTICES %d %d\n", N, 2 * N);
        for (int i = 0; i < N; ++i) { putBE(buf, (int32_t)1); putBE(buf, (int32_t)i); }
        flush();
        std::fprintf(fp, "\nPOINT_DATA %d\n", N);
        auto scalar = [&](const char* name, auto get) {
            std::fprintf(fp, "SCALARS %s float 1\nLOOKUP_TABLE default\n", name);
            for (int i = 0; i < N; ++i) putBE(buf, (float)get(i));
            flush();
            std::fprintf(fp, "\n");
        };
        scalar("damage", [&](int i) { return dmg[i]; });
        scalar("temperature_K", [&](int i) { return T[i]; });
        scalar("body", [&](int i) { return (double)body[i]; });
        scalar("fragment", [&](int i) { return (double)fragId[i]; });
        auto cS = [&](int i, int c) { double r, t, z; cyl(&sig[6 * (size_t)i], i, r, t, z); return (c == 0 ? r : c == 1 ? t : z) * 1e-6; };
        scalar("sigma_rr_MPa", [&](int i) { return cS(i, 0); });
        scalar("sigma_tt_MPa", [&](int i) { return cS(i, 1); });
        scalar("sigma_zz_MPa", [&](int i) { return cS(i, 2); });
        scalar("von_mises_MPa", [&](int i) { return vonMises(&sig[6 * (size_t)i]) * 1e-6; });
        std::fprintf(fp, "VECTORS displacement float\n");
        for (size_t k = 0; k < 3 * (size_t)N; ++k) putBE(buf, (float)u[k]);
        flush();
        std::fprintf(fp, "\nTENSORS stress_Pa float\n");
        for (int i = 0; i < N; ++i) {
            const float* s = &sig[6 * (size_t)i];
            const float t9[9] = {s[0], s[5], s[4], s[5], s[1], s[3], s[4], s[3], s[2]};
            for (float x : t9) putBE(buf, x);
        }
        flush();
        std::fclose(fp);
    }

    // full state for crackmesh.py and custom analysis (little-endian, documented header)
    void writeState(const std::string& fn, int step) const {
        FILE* fp = std::fopen(fn.c_str(), "wb");
        if (!fp) return;
        std::vector<uint32_t> mask(N, 0);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            uint32_t mk = 0;
            for (int64_t q = off[i]; q < off[i + 1]; ++q) {
                const int j = nbr[q];
                const int di = gi[j] - gi[i], dj = gj[j] - gj[i], dk = gk[j] - gk[i];
                if (std::abs(di) <= 1 && std::abs(dj) <= 1 && std::abs(dk) <= 1 && brk[q])
                    mk |= 1u << ((di + 1) * 9 + (dj + 1) * 3 + (dk + 1));
            }
            mask[i] = mk;
        }
        std::fprintf(fp, "PDSTATE2\n");
        std::fprintf(fp,
                     "{\"version\":2,\"build\":\"%s\",\"labels\":%s,\"N\":%d,\"nx\":%d,\"ny\":%d,\"nz\":%d,\"dx\":%.9e,\"delta\":%.9e,\"R\":%.9e,"
                     "\"nbody\":%d,\"step\":%d,\"time_h\":%.6f,\"q_Wpm\":%.6f,\"p_Pa\":%.6f,\"Tref\":%.4f,"
                     "\"arrays\":[\"gi:i4:1\",\"gj:i4:1\",\"gk:i4:1\",\"body:u1:1\",\"x0:f4:3\",\"u:f4:3\",\"T:f4:1\","
                     "\"damage:f4:1\",\"sigma:f4:6\",\"eps:f4:6\",\"fragment:i4:1\",\"nbmask:u4:1\"],"
                     "\"tensor_order\":\"xx,yy,zz,yz,xz,xy\",\"nbmask_bit\":\"(di+1)*9+(dj+1)*3+(dk+1)\"}\n",
                     PD_BUILD_ID, labelsJSON(p.labels).c_str(), N, nxy, nxy, nzTot, dx, delta, p.R, NBODY, step, tnow, qnow,
                     pnow, p.Tref);
        auto wr = [&](const void* d, size_t sz, size_t n) { std::fwrite(d, sz, n, fp); };
        wr(gi.data(), 4, N); wr(gj.data(), 4, N); wr(gk.data(), 4, N); wr(body.data(), 1, N);
        std::vector<float> tmp(3 * (size_t)N);
        for (size_t k = 0; k < 3 * (size_t)N; ++k) tmp[k] = (float)x0[k];
        wr(tmp.data(), 4, tmp.size());
        for (size_t k = 0; k < 3 * (size_t)N; ++k) tmp[k] = (float)u[k];
        wr(tmp.data(), 4, tmp.size());
        std::vector<float> t1(N);
        for (int i = 0; i < N; ++i) t1[i] = (float)T[i];
        wr(t1.data(), 4, N);
        wr(dmg.data(), 4, N);
        wr(sig.data(), 4, sig.size());
        wr(eps.data(), 4, eps.size());
        wr(fragId.data(), 4, N);
        wr(mask.data(), 4, N);
        std::fclose(fp);
    }

    // mid-plane radial profiles of every body (+ analytic elastic solution of a long cylinder)
    void writeProfile(const std::string& fn, double* rmsOut) const {
        FILE* fp = std::fopen(fn.c_str(), "w");
        if (!fp) return;
        std::fprintf(fp, "body,r_mm,n,T_K,u_r_um,damage,sigma_rr_MPa,sigma_tt_MPa,sigma_zz_MPa,sigma_vm_MPa,"
                         "eps_rr,eps_tt,eps_zz,n_stress,an_sigma_rr_MPa,an_sigma_tt_MPa,an_sigma_zz_MPa\n");
        const int nb = p.nr;
        const double Aan = p.alpha * p.E * dT0 / (4.0 * (1.0 - p.nu));
        double err = 0; int nerr = 0;
        for (int b = 0; b < NBODY; ++b) {
            const int kl = midLayer(b);
            std::vector<double> acc((size_t)nb * 12, 0.0);
            std::vector<int> cnt(nb, 0), cs(nb, 0);
            for (int i = bStart[b]; i < bStart[b] + bCount[b]; ++i) {
                if (gk[i] != kl) continue;
                const double x = x0[3 * (size_t)i], y = x0[3 * (size_t)i + 1], r = std::hypot(x, y);
                const int k = std::min(nb - 1, (int)(r / p.R * nb));
                double* a = &acc[(size_t)k * 12];
                a[0] += T[i];
                a[1] += r > 1e-12 ? (u[3 * (size_t)i] * x + u[3 * (size_t)i + 1] * y) / r : 0.0;
                a[2] += dmg[i];
                ++cnt[k];
                double rr, tt, zz, er, et, ez;
                cyl(&sig[6 * (size_t)i], i, rr, tt, zz);
                if (std::isnan(rr)) continue;
                cyl(&eps[6 * (size_t)i], i, er, et, ez);
                a[3] += rr; a[4] += tt; a[5] += zz; a[6] += vonMises(&sig[6 * (size_t)i]);
                a[7] += er; a[8] += et; a[9] += ez;
                ++cs[k];
            }
            for (int k = 0; k < nb; ++k) {
                if (!cnt[k]) continue;
                const double* a = &acc[(size_t)k * 12];
                const double r = (k + 0.5) / nb * p.R, xi = r * r / (p.R * p.R);
                const double ar = Aan * (xi - 1), at = Aan * (3 * xi - 1), az = 2 * Aan * (2 * xi - 1);
                const double c = cnt[k], s = std::max(1, cs[k]);
                std::fprintf(fp, "%d,%.5f,%d,%.3f,%.4f,%.5f,%.4f,%.4f,%.4f,%.4f,%.6e,%.6e,%.6e,%d,%.4f,%.4f,%.4f\n", b,
                             r * 1e3, cnt[k], a[0] / c, a[1] / c * 1e6, a[2] / c, a[3] / s * 1e-6, a[4] / s * 1e-6,
                             a[5] / s * 1e-6, a[6] / s * 1e-6, a[7] / s, a[8] / s, a[9] / s, cs[k], ar * 1e-6, at * 1e-6,
                             az * 1e-6);
                if (b == mainBody && cs[k]) { err += (a[4] / s - at) * (a[4] / s - at); ++nerr; }
            }
        }
        std::fclose(fp);
        if (rmsOut) *rmsOut = (nerr && Aan > 0) ? std::sqrt(err / nerr) / (2 * Aan) : NAN;
    }

    // damage vs angle on the mid-plane outer ring and on both end faces of every body
    void writeAngular(const std::string& fn) const {
        const int nbins = 180;
        FILE* fp = std::fopen(fn.c_str(), "w");
        if (!fp) return;
        std::fprintf(fp, "theta_deg");
        for (int b = 0; b < NBODY; ++b) std::fprintf(fp, ",b%d_mid,b%d_bottom,b%d_top", b, b, b);
        std::fprintf(fp, "\n");
        std::vector<float> P((size_t)NBODY * 3 * nbins, 0.f);
        for (int i = 0; i < N; ++i) {
            const int b = body[i];
            const double x = x0[3 * (size_t)i], y = x0[3 * (size_t)i + 1], r = std::hypot(x, y);
            if (r < 0.4 * p.R) continue;
            const int k = std::min(nbins - 1, (int)((std::atan2(y, x) + M_PI) / (2 * M_PI) * nbins));
            int which = -1;
            if (gk[i] == midLayer(b) && r > 0.75 * p.R) which = 0;
            else if (gk[i] <= bZ0[b] + 1) which = 1;
            else if (gk[i] >= bZ0[b] + bNz[b] - 2) which = 2;
            if (which < 0) continue;
            float& c = P[((size_t)b * 3 + which) * nbins + k];
            c = std::max(c, dmg[i]);
        }
        for (int k = 0; k < nbins; ++k) {
            std::fprintf(fp, "%.2f", -180.0 + (k + 0.5) * 360.0 / nbins);
            for (int b = 0; b < NBODY; ++b)
                for (int w = 0; w < 3; ++w) std::fprintf(fp, ",%.4f", P[((size_t)b * 3 + w) * nbins + k]);
            std::fprintf(fp, "\n");
        }
        std::fclose(fp);
    }

    void writeInterface(const std::string& fn) const {
        if (pairs.empty()) return;
        FILE* fp = std::fopen(fn.c_str(), "w");
        if (!fp) return;
        std::fprintf(fp, "iface,r_mm,theta_deg,gap_um,p_MPa,tau_MPa,slip_um,state,dmg_lower,dmg_upper\n");
        for (auto& c : pairs) {
            const double sl = std::hypot(c.sc[0], c.sc[1]);
            const int state = c.gap >= 0 ? 0 : (p.mu >= 0 && c.tn >= 0.999 * p.mu * c.pn ? 2 : 1);  // 0 open 1 stick 2 slip
            std::fprintf(fp, "%d,%.5f,%.3f,%.5f,%.5f,%.5f,%.5f,%d,%.4f,%.4f\n", c.iface, c.r * 1e3, c.th * 180.0 / M_PI,
                         c.gap * 1e6, c.pn * 1e-6, c.tn * 1e-6, sl * 1e6, state, dmg[c.i], dmg[c.j]);
        }
        std::fclose(fp);
    }

    void writeManifest(const std::string& fn) const {
        FILE* fp = std::fopen(fn.c_str(), "w");
        if (!fp) return;
        const Params& q = p;
        std::fprintf(fp, "{\n  \"schema\": \"pellet_pd.manifest/2\",\n  \"version\": \"%s\",\n  \"build\": \"%s\",\n"
                         "  \"labels\": %s,\n  \"units\": \"SI (m, Pa, K, W/m, J/m2); time in hours\",\n  \"command\": \"%s",
                     PD_VERSION, PD_BUILD_ID, labelsJSON(p.labels).c_str(), jesc(cmdline).c_str());
        std::fprintf(fp, "\",\n  \"threads\": %d,\n", omp_get_max_threads());
        std::fprintf(fp, "  \"params\": {\"R\":%.9e,\"H\":%.9e,\"dishR\":%.9e,\"dishD\":%.9e,\"chamfer\":%.9e,\"nr\":%d,"
                         "\"mfac\":%.6f,\"delta_in\":%.9e,\"jitter\":%.4f,\"stack\":\"%s\",\"E\":%.6e,\"nu\":%.5f,\"alpha\":%.6e,"
                         "\"Gc\":%.6f,\"sc_in\":%.6e,\"sigmaf\":%.6e,\"calib_delta\":%s,\"weibull\":%.4f,\"wcorr\":%.9e,"
                         "\"seed\":%.0f,\"ccomp\":%.4f,\"creep\":%s,\"tauRef\":%.6e,\"TtauRef\":%.3f,\"Qact\":%.6e,"
                         "\"lamv\":%.4f,\"qmax\":%.4f,\"kth\":%.5f,\"Tref\":%.4f,\"gsurf\":%.6e,\"history\":\"%s\","
                         "\"dqmax\":%.4f,\"dtmax\":%.4f,\"steps\":%d,\"pin\":\"%s\",\"pressure\":%.6e,\"mu\":%.5f,"
                         "\"cn\":%.4f,\"ctang\":%.4f,\"contact_rmin\":%.9e,\"maxit\":%d,\"maxcycles\":%d,\"kappa\":%.4f,"
                         "\"tol\":%.3e,\"sf\":%.4f,\"dstress\":%.3f,\"cover\":%.3f,\"dcut\":%.3f,\"minfrag\":%d,"
                         "\"onsetFrac\":%.4f,\"elastic\":%s,\"restart\":\"%s\",\"restart_body\":%d},\n",
                     q.R, q.H, q.dishR, q.dishD, q.chamfer, q.nr, q.mfac, q.delta, q.jitter, q.stack.c_str(), q.E, q.nu, q.alpha,
                     q.Gc, q.sc, q.sigmaf, q.calibDelta ? "true" : "false", q.weibull, q.wcorr, q.seed, q.ccomp,
                     q.creep ? "true" : "false", q.tauRef, q.TtauRef, q.Qact, q.lamv, q.qmax, q.kth, q.Tref, q.gsurf,
                     jesc(q.history).c_str(), q.dqmax, q.dtmax, q.steps, q.pin.c_str(), q.pressure, q.mu, q.cn, q.ctang,
                     q.contactRmin, q.maxit, q.maxcycles, q.kappa, q.tol, q.sf, q.dstress, q.cover, q.dcut, q.minfrag,
                     q.onsetFrac, q.elastic ? "true" : "false", jesc(q.restartIn).c_str(), q.restartBody);
        std::fprintf(fp, "  \"derived\": {\"dx\":%.9e,\"delta\":%.9e,\"m_ratio\":%.5f,\"V\":%.9e,\"K\":%.6e,\"mu\":%.6e,"
                         "\"lambda\":%.6e,\"sc0\":%.6e,\"sc_from_Gc\":%.6e,\"sigma_from_sc_MPa\":%.4f,\"C_LPS\":%.6e,"
                         "\"irwin_length\":%.6e,\"kn\":%.6e,\"kt\":%.6e,\"Wbulk\":%.6e},\n",
                     dx, delta, delta / dx, V, K, mu, lamL, sc0, scG, sc0 * p.E * 1e-6, Ccal,
                     p.E * p.Gc / std::pow(sc0 * p.E, 2), kn, kt, Wbulk);
        std::fprintf(fp, "  \"counts\": {\"N\":%d,\"NB_directed\":%lld,\"avg_family\":%.3f,\"nx\":%d,\"nz\":%d,"
                         "\"contact_pairs\":%zu,\"top_particles\":%d,\"bodies\":[",
                     N, (long long)NB, N ? (double)NB / N : 0.0, nxy, nzTot, pairs.size(), nTop);
        for (int b = 0; b < NBODY; ++b)
            std::fprintf(fp, "%s{\"id\":%d,\"h_rel\":%.4f,\"H\":%.9e,\"nz\":%d,\"z0_layer\":%d,\"N\":%d}", b ? "," : "", b,
                         hb[b], bH[b], bNz[b], bZ0[b], bCount[b]);
        std::fprintf(fp, "]},\n  \"schedule\": [");
        for (size_t s = 0; s < sched.size(); ++s)
            std::fprintf(fp, "%s[%.6f,%.6f,%.4f,%.4f,%d]", s ? "," : "", sched[s].t, sched[s].dt, sched[s].q, sched[s].p,
                         sched[s].knot ? 1 : 0);
        std::fprintf(fp, "],\n  \"schedule_columns\": [\"time_h\",\"dt_h\",\"q_Wpm\",\"p_Pa\",\"is_knot\"],\n");
        std::fprintf(fp, "  \"setup_time_s\": {\"kdtree\":%.4f,\"bonds\":%.4f}\n}\n", tKD, tBond);
        std::fclose(fp);
    }
};

// ----------------------------------------------------------------------------
// Checkpoint / resume (whole model, any number of bodies) -- for Spot VMs.
// Written atomically (tmp + rename) after every --ckpt-every completed steps.
// ----------------------------------------------------------------------------
struct RunAgg {
    int stepDone = 0;
    double firstCrackQ = -1, onsetQ = -1, onsetT = -1, maxArea = 0, tSolve = 0, peakFragQ = -1;
    long long itTotal = 0;
    std::string lastFrag = "[]", peakFrag = "[]";
};

static void writeStr(FILE* fp, const std::string& s) {
    const uint32_t n = (uint32_t)s.size();
    std::fwrite(&n, 4, 1, fp);
    std::fwrite(s.data(), 1, n, fp);
}
static bool readStr(FILE* fp, std::string& s) {
    uint32_t n;
    if (std::fread(&n, 4, 1, fp) != 1 || n > (1u << 26)) return false;
    s.resize(n);
    return std::fread(&s[0], 1, n, fp) == n;
}

static void writeCheckpoint(const Model& M, const std::string& fn, const RunAgg& a) {
    const std::string tmp = fn + ".tmp";
    FILE* fp = std::fopen(tmp.c_str(), "wb");
    if (!fp) { std::fprintf(stderr, "WARNING: cannot write checkpoint %s\n", tmp.c_str()); return; }
    const char magic[8] = {'P', 'D', 'C', 'K', 'P', '0', '2', 0};
    std::fwrite(magic, 1, 8, fp);
    const int32_t h[3] = {M.N, a.stepDone, M.p.creep ? 1 : 0};
    std::fwrite(h, 4, 3, fp);
    const int64_t h64[3] = {M.NB, (int64_t)M.pairs.size(), a.itTotal};
    std::fwrite(h64, 8, 3, fp);
    const double d[7] = {M.tnow, a.firstCrackQ, a.onsetQ, a.onsetT, a.maxArea, a.tSolve, a.peakFragQ};
    std::fwrite(d, 8, 7, fp);
    std::fwrite(M.u.data(), 8, M.u.size(), fp);
    std::fwrite(M.brk.get(), 1, M.NB, fp);
    if (M.p.creep) std::fwrite(M.eb.get(), 4, M.NB, fp);
    for (auto& c : M.pairs) std::fwrite(c.sc, 8, 2, fp);
    writeStr(fp, a.lastFrag);
    writeStr(fp, a.peakFrag);
    const bool ok = std::fflush(fp) == 0;
    std::fclose(fp);
    if (ok) std::rename(tmp.c_str(), fn.c_str());
}

static bool readCheckpoint(Model& M, const std::string& fn, RunAgg& a) {
    FILE* fp = std::fopen(fn.c_str(), "rb");
    if (!fp) return false;
    char magic[8];
    int32_t h[3];
    int64_t h64[3];
    double d[7];
    bool ok = std::fread(magic, 1, 8, fp) == 8 && std::memcmp(magic, "PDCKP02", 7) == 0 && std::fread(h, 4, 3, fp) == 3 &&
              std::fread(h64, 8, 3, fp) == 3 && std::fread(d, 8, 7, fp) == 7;
    if (ok && (h[0] != M.N || h64[0] != M.NB || h64[1] != (int64_t)M.pairs.size() || h[2] != (M.p.creep ? 1 : 0))) {
        std::fprintf(stderr, "WARNING: checkpoint does not match this model; starting from scratch\n");
        std::fclose(fp);
        return false;
    }
    ok = ok && std::fread(M.u.data(), 8, M.u.size(), fp) == M.u.size() && std::fread(M.brk.get(), 1, M.NB, fp) == (size_t)M.NB;
    if (ok && M.p.creep) ok = std::fread(M.eb.get(), 4, M.NB, fp) == (size_t)M.NB;
    for (auto& c : M.pairs) if (ok) ok = std::fread(c.sc, 8, 2, fp) == 2;
    ok = ok && readStr(fp, a.lastFrag) && readStr(fp, a.peakFrag);
    std::fclose(fp);
    if (!ok) { std::fprintf(stderr, "ERROR: corrupt checkpoint %s\n", fn.c_str()); std::exit(3); }
    a.stepDone = h[1]; a.itTotal = h64[2];
    M.tnow = d[0]; a.firstCrackQ = d[1]; a.onsetQ = d[2]; a.onsetT = d[3]; a.maxArea = d[4]; a.tSolve = d[5]; a.peakFragQ = d[6];
    return true;
}

// keep header + rows whose first integer field (step) <= maxStep (drop rows written after the checkpoint)
static void truncateCsv(const std::string& fn, int maxStep) {
    FILE* fp = std::fopen(fn.c_str(), "r");
    if (!fp) return;
    std::vector<std::string> keep;
    char line[65536];
    bool first = true;
    while (std::fgets(line, sizeof line, fp)) {
        if (first) { keep.push_back(line); first = false; continue; }
        if (std::atoi(line) <= maxStep) keep.push_back(line);
    }
    std::fclose(fp);
    fp = std::fopen(fn.c_str(), "w");
    if (!fp) return;
    for (auto& l : keep) std::fputs(l.c_str(), fp);
    std::fclose(fp);
}

static bool fileExists(const std::string& fn) { struct stat st; return stat(fn.c_str(), &st) == 0; }

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    Model M;
    if (!parseArgs(argc, argv, M.p, M.cmdline)) { usage(); return 1; }
    Params& P = M.p;
    std::signal(SIGTERM, onSignal);
    std::signal(SIGINT, onSignal);
    mkdir(P.out.c_str(), 0755);
    const double tstart = omp_get_wtime();
    std::printf("=== %s | threads %d ===\n", PD_VERSION, omp_get_max_threads());
    M.init();
    std::printf("bodies %d (stack %s) | dx %.4f mm, delta %.4f mm (m=%.2f) | particles %d\n", M.NBODY, P.stack.c_str(),
                M.dx * 1e3, M.delta * 1e3, M.delta / M.dx, M.N);
    std::printf("s_c0 %.4e (G_c-based %.4e) -> strength ~ %.1f MPa | Weibull m %g, wcorr %.3g mm | creep %s\n", M.sc0, M.scG,
                M.sc0 * P.E * 1e-6, P.weibull, P.wcorr * 1e3, P.creep ? "on" : "off");

    if (P.dryrun) {
        KDTree tree(M.x0.data(), M.N);
        std::vector<int> cnt(M.N);
        M.countBonds(tree, cnt);
        int64_t nb = 0;
        for (int c : cnt) nb += c;
        const double bytesB = 17.0 + (P.creep ? 6.0 : 0.0);
        const double mem = nb * bytesB + M.N * 330.0;
        std::printf("DRYRUN  particles %d  bonds %lld  memory ~ %.2f GB  ~%.3f s per ADR iteration at %.1f G bond-evals/s\n",
                    M.N, (long long)nb, mem / 1073741824.0, 2.0 * nb / (P.gbes * 1e9), P.gbes);
        FILE* fp = std::fopen((P.out + "/dryrun.json").c_str(), "w");
        if (fp) {
            std::fprintf(fp, "{\"N\":%d,\"NB\":%lld,\"mem_GB\":%.4f,\"s_per_iter\":%.6f,\"dx\":%.9e,\"delta\":%.9e,\"nr\":%d}\n",
                         M.N, (long long)nb, mem / 1073741824.0, 2.0 * nb / (P.gbes * 1e9), M.dx, M.delta, P.nr);
            std::fclose(fp);
        }
        return 0;
    }

    M.buildBonds();
    M.buildSchedule();
    std::printf("bonds %lld (avg family %.1f) | contact pairs %zu | load steps %zu | setup %.2f s\n", (long long)M.NB,
                (double)M.NB / M.N, M.pairs.size(), M.sched.size(), M.tKD + M.tBond);
    M.writeManifest(P.out + "/manifest.json");

    // ---- resume? ----
    RunAgg agg;
    const std::string ckpt = P.out + "/checkpoint.bin";
    bool resumed = false;
    if (P.resume && fileExists(ckpt) && readCheckpoint(M, ckpt, agg)) {
        resumed = true;
        std::printf("RESUME from step %d (t = %.4f h)\n", agg.stepDone, M.tnow);
        for (const char* f : {"/history.csv", "/fragments.csv", "/cycles.csv"}) truncateCsv(P.out + f, agg.stepDone);
    }
    const char* mode = resumed ? "a" : "w";
    M.cyclog = std::fopen((P.out + "/cycles.csv").c_str(), mode);
    if (M.cyclog && !resumed) std::fprintf(M.cyclog, "step,cycle,max_ratio,bonds_broken,relax_iters,residual\n");
    FILE* hist = std::fopen((P.out + "/history.csv").c_str(), mode);
    FILE* frag = std::fopen((P.out + "/fragments.csv").c_str(), mode);
    if (frag && !resumed)
        std::fprintf(frag, "step,time_h,q_Wpm,body,frag_id,n_particles,volume_mm3,d_eq_mm,cx_mm,cy_mm,cz_mm,"
                           "rmin_mm,rmax_mm,zmin_mm,zmax_mm\n");
    if (hist && !resumed) {
        std::fprintf(hist, "step,time_h,dt_h,q_Wpm,p_Pa,is_knot,T_centre_K,T_surface_K,adr_iters,break_cycles,residual,"
                           "scale,new_broken_pairs,broken_pairs,crack_area_mm2,mean_damage,max_damage,"
                           "stt_p50_MPa,stt_p99_MPa,stt_max_MPa,szz_p99_MPa,svm_p99_MPa,n_stress_valid,"
                           "max_contact_p_MPa,mean_contact_p_MPa,contact_frac_closed,contact_frac_slip,mean_slip_um,"
                           "wall_s");
        for (int b = 0; b < M.NBODY; ++b)
            std::fprintf(hist, ",b%d_crack_area_mm2,b%d_broken_pairs,b%d_radial_cracks,b%d_ur_mid_um,b%d_ur_end_um,b%d_dH_um",
                         b, b, b, b, b, b);
        std::fprintf(hist, "\n");
    }

    const int nsteps = (int)M.sched.size();
    double Ab[MAXB];
    int64_t PBb[MAXB];
    double qpk = 0;
    for (auto& s : M.sched) qpk = std::max(qpk, s.q);
    const double onsetA = P.onsetFrac * P.R * M.bH[M.mainBody];

    auto fragJSON = [&](Model::FragStats* st) {
        std::string s = "[";
        char buf[512];
        for (int b = 0; b < M.NBODY; ++b) {
            std::snprintf(buf, sizeof buf,
                          "%s{\"body\":%d,\"n\":%d,\"vol_total_mm3\":%.6f,\"vol_mean_mm3\":%.6f,\"vol_median_mm3\":%.6f,"
                          "\"deq_mean_mm\":%.6f,\"deq_volweighted_mm\":%.6f,\"largest_frac\":%.6f,\"Leq_mm\":%.6f}",
                          b ? "," : "", b, st[b].n, st[b].vtot * 1e9, st[b].vmean * 1e9, st[b].vmed * 1e9,
                          st[b].deqMean * 1e3, st[b].deqVW * 1e3, st[b].largestFrac, st[b].Leq * 1e3);
            s += buf;
        }
        return s + "]";
    };

    std::printf("\n%5s %8s %7s %6s %7s %7s %6s %9s %10s %9s %7s %7s\n", "step", "t[h]", "q'kW/m", "pMPa", "Tc[K]", "iters",
                "cyc", "resid", "broken", "A[mm2]", "cracks", "t[s]");
    for (int s = agg.stepDone; s < nsteps; ++s) {
        const LoadStep& L = M.sched[s];
        M.curStep = s + 1;
        M.prepareStep(L);
        const double t0 = omp_get_wtime();
        auto info = M.solveLoadStep();
        if (gStop) {   // preempted mid-step: discard this step, resume from the last checkpoint
            std::fprintf(stderr, "signal received during step %d: exiting (resume with --resume)\n", s + 1);
            if (hist) std::fclose(hist);
            if (frag) std::fclose(frag);
            if (M.cyclog) std::fclose(M.cyclog);
            return 75;
        }
        M.commitCreep();
        const double t1 = omp_get_wtime();
        agg.tSolve += t1 - t0;
        agg.itTotal += info.iters;

        M.computeStressStrain();
        M.crackArea(Ab, PBb);
        double Atot = 0; int64_t PBt = 0;
        for (int b = 0; b < M.NBODY; ++b) { Atot += Ab[b]; PBt += PBb[b]; }
        agg.maxArea = std::max(agg.maxArea, Atot);
        if (agg.firstCrackQ < 0 && PBt > 0) agg.firstCrackQ = L.q;
        if (agg.onsetQ < 0 && Ab[M.mainBody] >= onsetA) { agg.onsetQ = L.q; agg.onsetT = L.t; }

        double meanD = 0, maxD = 0;
        std::vector<double> stt, szz, svm;
        stt.reserve(M.N);
        for (int i = 0; i < M.N; ++i) {
            meanD += M.dmg[i]; maxD = std::max(maxD, (double)M.dmg[i]);
            double r, t, z;
            M.cyl(&M.sig[6 * (size_t)i], i, r, t, z);
            if (std::isnan(t)) continue;
            stt.push_back(t * 1e-6); szz.push_back(z * 1e-6); svm.push_back(Model::vonMises(&M.sig[6 * (size_t)i]) * 1e-6);
        }
        meanD /= std::max(1, M.N);
        double pmax = 0, psum = 0, slip = 0; int nclosed = 0, nslip = 0;
        for (auto& c : M.pairs) {
            pmax = std::max(pmax, c.pn); psum += c.pn; slip += std::hypot(c.sc[0], c.sc[1]);
            if (c.gap < 0) { ++nclosed; if (P.mu >= 0 && c.tn >= 0.999 * P.mu * c.pn) ++nslip; }
        }
        const double np = (double)std::max<size_t>(1, M.pairs.size());
        const int cracksMain = M.countRadialCracks(M.mainBody);
        std::printf("%5d %8.3f %7.2f %6.2f %7.1f %7d %6d %9.2e %10lld %9.3f %7d %7.1f\n", s + 1, L.t, L.q * 1e-3, L.p * 1e-6,
                    M.Ts + M.dT0, info.iters, info.cycles, info.res, (long long)PBt, Atot * 1e6, cracksMain, t1 - t0);
        std::fflush(stdout);
        if (hist) {
            std::fprintf(hist, "%d,%.6f,%.6f,%.4f,%.4f,%d,%.3f,%.3f,%d,%d,%.4e,%.4e,%lld,%lld,%.6f,%.6f,%.5f,%.4f,%.4f,%.4f,%.4f,"
                               "%.4f,%zu,%.5f,%.5f,%.5f,%.5f,%.5f,%.3f",
                         s + 1, L.t, L.dt, L.q, L.p, L.knot ? 1 : 0, M.Ts + M.dT0, M.Ts, info.iters, info.cycles, info.res,
                         info.scale, (long long)(info.broken / 2), (long long)PBt, Atot * 1e6, meanD, maxD,
                         percentile(stt, 0.5), percentile(stt, 0.99),
                         stt.empty() ? NAN : *std::max_element(stt.begin(), stt.end()), percentile(szz, 0.99),
                         percentile(svm, 0.99), stt.size(), pmax * 1e-6, psum / np * 1e-6, nclosed / np,
                         nclosed ? (double)nslip / nclosed : 0.0, slip / np * 1e6, t1 - t0);
            for (int b = 0; b < M.NBODY; ++b) {
                double urm, ure, dH;
                M.dims(b, urm, ure, dH);
                std::fprintf(hist, ",%.6f,%lld,%d,%.5f,%.5f,%.5f", Ab[b] * 1e6, (long long)PBb[b], M.countRadialCracks(b),
                             urm * 1e6, ure * 1e6, dH * 1e6);
            }
            std::fprintf(hist, "\n");
            std::fflush(hist);
        }

        const bool doOut = L.knot || ((s + 1) % P.every == 0) || (s + 1 == nsteps);
        if (doOut) {
            Model::FragStats st[MAXB];
            auto frs = M.fragments(st, Ab);
            if (frag) {
                for (auto& F : frs)
                    std::fprintf(frag, "%d,%.6f,%.4f,%d,%d,%d,%.6f,%.6f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", s + 1, L.t, L.q,
                                 F.body, F.id, F.n, F.vol * 1e9, F.deq * 1e3, F.cx * 1e3, F.cy * 1e3, F.cz * 1e3,
                                 F.rmin * 1e3, F.rmax * 1e3, F.zmin * 1e3, F.zmax * 1e3);
                std::fflush(frag);
            }
            agg.lastFrag = fragJSON(st);
            if (L.q >= qpk - 1e-9) { agg.peakFrag = agg.lastFrag; agg.peakFragQ = L.q; }
            char name[64];
            double rms = NAN;
            std::snprintf(name, sizeof name, "/profile_%04d.csv", s + 1);
            M.writeProfile(P.out + name, &rms);
            if (PBt == 0 && M.NBODY == 1 && !std::isnan(rms))
                std::printf("      mid-plane hoop stress vs analytic: RMS error = %.1f %% of surface value\n", 100 * rms);
            std::snprintf(name, sizeof name, "/angular_%04d.csv", s + 1);
            M.writeAngular(P.out + name);
            std::snprintf(name, sizeof name, "/interface_%04d.csv", s + 1);
            M.writeInterface(P.out + name);
            if (!P.nostate) { std::snprintf(name, sizeof name, "/state_%04d.bin", s + 1); M.writeState(P.out + name, s + 1); }
            if (!P.novtk) { std::snprintf(name, sizeof name, "/pellet_%04d.vtk", s + 1); M.writeVTK(P.out + name); }
        }
        agg.stepDone = s + 1;
        if (P.ckptEvery > 0 && ((s + 1) % P.ckptEvery == 0 || s + 1 == nsteps)) writeCheckpoint(M, ckpt, agg);
    }
    if (hist) std::fclose(hist);
    if (frag) std::fclose(frag);
    if (M.cyclog) std::fclose(M.cyclog);
    if (!P.restartOut.empty()) M.writeRestart(P.restartOut);

    const double ttot = omp_get_wtime() - tstart;
    FILE* fp = std::fopen((P.out + "/summary.json").c_str(), "w");
    if (fp) {
        M.crackArea(Ab, PBb);
        std::fprintf(fp, "{\n  \"schema\": \"pellet_pd.summary/2\",\n  \"version\": \"%s\",\n  \"build\": \"%s\",\n"
                         "  \"labels\": %s,\n  \"complete\": true,\n  \"resumed\": %s,\n  \"N\": %d,\n"
                         "  \"NB_directed\": %lld,\n  \"steps\": %d,\n",
                     PD_VERSION, PD_BUILD_ID, labelsJSON(P.labels).c_str(), resumed ? "true" : "false", M.N, (long long)M.NB, nsteps);
        std::fprintf(fp, "  \"dx\": %.9e,\n  \"delta\": %.9e,\n  \"sc0\": %.6e,\n  \"q_peak_Wpm\": %.4f,\n", M.dx, M.delta, M.sc0, qpk);
        std::fprintf(fp, "  \"first_bond_failure_q_Wpm\": %.4f,\n  \"onset_q_Wpm\": %.4f,\n  \"onset_time_h\": %.6f,\n"
                         "  \"onset_area_threshold_mm2\": %.6f,\n",
                     agg.firstCrackQ, agg.onsetQ, agg.onsetT, onsetA * 1e6);
        std::fprintf(fp, "  \"crack_area_final_mm2\": [");
        for (int b = 0; b < M.NBODY; ++b) std::fprintf(fp, "%s%.6f", b ? "," : "", Ab[b] * 1e6);
        std::fprintf(fp, "],\n  \"broken_pairs_final\": [");
        for (int b = 0; b < M.NBODY; ++b) std::fprintf(fp, "%s%lld", b ? "," : "", (long long)PBb[b]);
        std::fprintf(fp, "],\n  \"crack_area_max_total_mm2\": %.6f,\n", agg.maxArea * 1e6);
        std::fprintf(fp, "  \"radial_cracks_final\": [");
        for (int b = 0; b < M.NBODY; ++b) std::fprintf(fp, "%s%d", b ? "," : "", M.countRadialCracks(b));
        std::fprintf(fp, "],\n  \"fragments_final\": %s,\n  \"fragments_at_peak\": %s,\n  \"fragments_at_peak_q_Wpm\": %.4f,\n",
                     agg.lastFrag.c_str(), agg.peakFrag.c_str(), agg.peakFragQ);
        std::fprintf(fp, "  \"adr_iterations_total\": %lld,\n  \"solver_time_s\": %.3f,\n  \"total_time_s\": %.3f,\n"
                         "  \"throughput_Mbond_per_s\": %.3f,\n  \"threads\": %d\n}\n",
                     agg.itTotal, agg.tSolve, ttot, agg.tSolve > 0 ? 2.0 * M.NB * agg.itTotal / agg.tSolve * 1e-6 : 0.0,
                     omp_get_max_threads());
        std::fclose(fp);
    }
    std::printf("\nsummary: crack area %.3f mm2, onset q' %.2f kW/m, solver %.1f s, total %.1f s -> %s/\n", agg.maxArea * 1e6,
                agg.onsetQ * 1e-3, agg.tSolve, ttot, P.out.c_str());
    return 0;
}
