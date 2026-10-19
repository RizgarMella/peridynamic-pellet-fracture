// =============================================================================
//  pellet_pd.cpp  --  Peridynamic simulation of thermal cracking in a nuclear
//                     fuel (UO2) pellet.  Single file, C++17 + OpenMP.
// -----------------------------------------------------------------------------
//  Physics
//   * 3D ordinary state-based peridynamics, Linear Peridynamic Solid (LPS,
//     Silling et al. 2007) -> arbitrary Poisson ratio (UO2: nu ~ 0.32), unlike
//     bond-based PD which is locked to nu = 1/4.
//   * Thermal expansion enters through the bond extension state:
//         e_mech = |Y| - |X| (1 + alpha * dT_bond),  dT_bond = (T_i+T_j)/2 - Tref
//     so a uniform temperature rise is stress free.
//   * Temperature: steady radial profile of a pellet at linear heat rate q'
//         T(r) = Ts + q'/(4 pi k) (1 - r^2/R^2)
//     ramped linearly from zero power to q'max over the load steps.
//   * Fracture: a bond breaks irreversibly when its mechanical stretch exceeds
//     a critical stretch s_c derived from the fracture energy G_c for LPS
//     (Foster, Silling & Chen 2011):
//         s_c = sqrt( G_c / ((3 mu + (3/4)^4 (K - 5 mu / 3)) delta) )
//     with optional Weibull scatter per bond (symmetric, deterministic hash)
//     to mimic microstructural variability and break lattice symmetry.
//   * Quasi-static equilibrium per load step via Adaptive Dynamic Relaxation
//     (Kilic & Madenci 2010), converged on the max-norm residual.  Bonds are
//     broken only at equilibrium and only the most critical ones per cycle
//     (stretch ratio >= kappa * max ratio), then the body is re-relaxed;
//     repeated until no bond exceeds s_c at that power.  This captures
//     unstable crack growth at fixed load and lets damage localise into
//     discrete radial cracks.  (Breaking inside the ADR iterations, or all
//     overstretched bonds at once, gives spurious diffuse damage.)
//   * Stress / strain: non-local deformation gradient from intact bonds
//         F = (sum w Y (x) X V)(sum w X (x) X V)^-1
//     small strain eps = sym(F) - I; elastic strain = eps - eps_th, with eps_th
//     fitted over the same family from bond-mean temperatures,
//     Hooke's law -> Cauchy stress (reported in Cartesian + cylindrical).
//     Crack debris (damage > dstress) and damaged particles with too little
//     intact family left (< cover * full family) report NaN stress.
//
//  Numerics / performance
//   * Particles on a cubic lattice; bonds found ONCE with a KD-tree radius
//     search (parallel queries) and stored in CSR (compressed row) arrays.
//   * Partial-volume correction for bonds crossing the horizon boundary.
//   * Force evaluation is a gather over each particle's own bond list:
//     no atomics, no locks, embarrassingly parallel with OpenMP.  Bond
//     breaking is decided identically from both ends (bitwise-symmetric
//     arithmetic), so bond i->j and j->i always break together.
//   * NUMA-friendly first-touch initialisation of the large bond arrays.
//
//  Build
//     g++ -O3 -march=native -fopenmp -std=c++17 pellet_pd.cpp -o pellet_pd
//     (clang++ works too; without -fopenmp it runs serially)
//
//  Run
//     OMP_NUM_THREADS=16 OMP_PROC_BIND=close ./pellet_pd --nr 20 --steps 40
//     ./pellet_pd --elastic --steps 1      # verification vs analytic solution
//     ./pellet_pd --help                   # all options (SI units)
//     --kappa 0.9..0.95 gives sharper cracks at higher cost; --verbose shows
//     every break/relax cycle.
//
//  Verification (built in): with --elastic, mid-plane stresses of a long
//  pellet are compared with the analytic solid-cylinder solution
//  (generalised plane strain, parabolic T).  Measured hoop-stress RMS error:
//  6.4 % (nr=8), 4.5 % (nr=12), 4.4 % (nr=16); the remainder is the
//  peridynamic surface ("skin") effect in the outermost particle layer.
//
//  Known limitations / notes
//   * Temperature is prescribed (analytic steady profile); cracks do not
//     change conduction.  Swap setTemperature() for a PD heat solver if needed.
//   * s_c from G_c scales as 1/sqrt(delta): coarse horizons crack at lower
//     power.  Use --sc to impose a strength-based critical stretch instead.
//   * Cubic lattice gives mild preferred crack directions (45/90 deg) at
//     coarse resolution; Weibull scatter and finer nr reduce this.
//   * Shared-memory (OpenMP) only; the CSR gather design maps directly onto
//     MPI domain decomposition with ghost layers of width delta.
//
//  Output (in --out directory, default pd_out/)
//     pellet_XXXX.vtk   legacy binary VTK point cloud for ParaView: damage,
//                       temperature, displacement, stress tensor, sigma_rr,
//                       sigma_tt, sigma_zz, von Mises, strains, dilatation
//                       (use "Point Gaussian" or "Warp By Vector")
//     profile_XXXX.csv  mid-plane radial profiles (PD vs analytic elastic)
//     history.csv       per-load-step summary
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

// ----------------------------------------------------------------------------
// Parameters (SI units)
// ----------------------------------------------------------------------------
struct Params {
    // geometry / discretisation
    double R = 4.1e-3;      // pellet radius [m]
    double H = 9.8e-3;      // pellet height [m]
    int    nr = 16;         // particles across the radius
    double mfac = 3.015;    // horizon delta = mfac * dx
    // UO2 material
    double E = 200e9;       // Young's modulus [Pa]
    double nu = 0.32;       // Poisson ratio
    double alpha = 1.0e-5;  // thermal expansion coefficient [1/K]
    double Gc = 10.0;       // fracture energy [J/m^2]  (K_IC ~ 1.5 MPa sqrt(m))
    double sc = 0.0;        // >0: critical stretch given directly (overrides Gc)
    double dstress = 0.5;   // damage above which stress is not reported (debris)
    double cover = 0.35;    // damaged particles with intact family < cover * full family: no stress
    double weibull = 12.0;  // Weibull modulus for s_c scatter (<=0: none)
    double seed = 12345;    // RNG seed for s_c scatter
    // thermal loading
    double qmax = 25e3;     // final linear heat rate [W/m]
    double kth = 3.0;       // thermal conductivity [W/m/K]
    double Tref = 293.15;   // stress-free reference temperature [K]
    double Tsurf = 773.15;  // pellet surface temperature at full power [K]
    // solver
    int    steps = 40;      // load steps (power ramp)
    int    maxit = 6000;    // max ADR iterations per relaxation
    int    maxcycles = 20000; // max break/relax cycles per load step
    double kappa = 0.8;     // break bonds with ratio >= kappa * max ratio per cycle
    double tol = 1e-3;      // ADR tolerance: max|f_i| / max|f_thermal_i|
    double sf = 1.0;        // ADR fictitious-mass safety factor
    // output
    std::string out = "pd_out";
    int    every = 5;       // write VTK/profile every N load steps (and last)
    bool   elastic = false; // disable fracture (verification)
    bool   novtk = false;
    bool   verbose = false; // print break/relax cycles
};

static void usage() {
    std::printf(
        "pellet_pd - peridynamic thermal cracking of a fuel pellet (SI units)\n"
        "  --R %%g        pellet radius [m]                  (4.1e-3)\n"
        "  --H %%g        pellet height [m]                  (9.8e-3)\n"
        "  --nr %%d       particles across radius            (16)\n"
        "  --mfac %%g     horizon / spacing                  (3.015)\n"
        "  --E --nu --alpha --Gc   elastic/thermal/fracture properties\n"
        "  --sc %%g       critical stretch, overrides Gc     (0 = from Gc)\n"
        "  --dstress %%g  max damage for stress reporting    (0.5)\n"
        "  --cover %%g    min intact-family coverage, damaged (0.35)\n"
        "  --weibull %%g  Weibull modulus of s_c (<=0 off)   (12)\n"
        "  --seed %%g     scatter seed                       (12345)\n"
        "  --qmax %%g     final linear heat rate [W/m]       (25e3)\n"
        "  --kth %%g      conductivity [W/m/K]               (3)\n"
        "  --Tref %%g     stress-free temperature [K]        (293.15)\n"
        "  --Tsurf %%g    surface temperature at qmax [K]    (773.15)\n"
        "  --steps %%d    load steps                         (40)\n"
        "  --maxit %%d    max ADR iterations per relaxation  (6000)\n"
        "  --maxcycles %%d max break/relax cycles per step   (20000)\n"
        "  --kappa %%g    break ratio >= kappa*max per cycle (0.8; 1 = sharpest, 0 = all at once)\n"
        "  --tol %%g      ADR tolerance, max-norm relative   (1e-3)\n"
        "  --sf %%g       ADR mass safety factor             (1.0)\n"
        "  --out %%s      output directory                   (pd_out)\n"
        "  --every %%d    output interval in load steps      (5)\n"
        "  --elastic     no bond breaking (verification)\n"
        "  --novtk       skip VTK output\n"
        "  --verbose     print every break/relax cycle\n");
}

static bool parseArgs(int argc, char** argv, Params& p) {
    std::map<std::string, double*> dbl = {
        {"R", &p.R}, {"H", &p.H}, {"mfac", &p.mfac}, {"E", &p.E}, {"nu", &p.nu},
        {"alpha", &p.alpha}, {"Gc", &p.Gc}, {"sc", &p.sc}, {"dstress", &p.dstress}, {"cover", &p.cover}, {"weibull", &p.weibull}, {"seed", &p.seed},
        {"qmax", &p.qmax}, {"kth", &p.kth}, {"Tref", &p.Tref}, {"Tsurf", &p.Tsurf},
        {"tol", &p.tol}, {"sf", &p.sf}, {"kappa", &p.kappa}};
    std::map<std::string, int*> itg = {
        {"nr", &p.nr}, {"steps", &p.steps}, {"maxit", &p.maxit}, {"maxcycles", &p.maxcycles}, {"every", &p.every}};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        if (a.rfind("--", 0) != 0) { std::fprintf(stderr, "bad argument: %s\n", a.c_str()); return false; }
        std::string key = a.substr(2), val;
        if (key == "elastic") { p.elastic = true; continue; }
        if (key == "novtk")   { p.novtk = true; continue; }
        if (key == "verbose") { p.verbose = true; continue; }
        auto eq = key.find('=');
        if (eq != std::string::npos) { val = key.substr(eq + 1); key = key.substr(0, eq); }
        else if (i + 1 < argc) val = argv[++i];
        else { std::fprintf(stderr, "missing value for %s\n", a.c_str()); return false; }
        try {
            if (dbl.count(key)) *dbl[key] = std::stod(val);
            else if (itg.count(key)) *itg[key] = std::stoi(val);
            else if (key == "out") p.out = val;
            else { std::fprintf(stderr, "unknown option --%s\n", key.c_str()); return false; }
        } catch (...) { std::fprintf(stderr, "bad value for --%s: %s\n", key.c_str(), val.c_str()); return false; }
    }
    return true;
}

// ----------------------------------------------------------------------------
// KD-tree (bounding-box nodes, median split on widest axis) for radius search
// ----------------------------------------------------------------------------
class KDTree {
public:
    KDTree(const double* pts, int n, int leaf = 12) : P_(pts), leaf_(leaf), idx_(n) {
        std::iota(idx_.begin(), idx_.end(), 0);
        nodes_.reserve(4 * (n / leaf + 1));
        build(0, n);
    }
    // calls cb(j, dist2) for every point with |p_j - q|^2 <= r2
    template <class CB>
    void radius(const double* q, double r2, CB&& cb) const {
        int stack[256];
        int sp = 0;
        stack[sp++] = 0;
        while (sp) {
            const Node& nd = nodes_[stack[--sp]];
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                double v = q[d] < nd.lo[d] ? nd.lo[d] - q[d] : (q[d] > nd.hi[d] ? q[d] - nd.hi[d] : 0.0);
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

template <class T>
static std::unique_ptr<T[]> uninit(size_t n) { return std::unique_ptr<T[]>(new T[n]); }  // no touch -> NUMA first-touch later

static inline bool inv3(const double A[9], double Ai[9], double& det) {
    const double c00 = A[4] * A[8] - A[5] * A[7];
    const double c01 = A[5] * A[6] - A[3] * A[8];
    const double c02 = A[3] * A[7] - A[4] * A[6];
    det = A[0] * c00 + A[1] * c01 + A[2] * c02;
    if (std::fabs(det) < 1e-300) return false;
    const double id = 1.0 / det;
    Ai[0] = c00 * id; Ai[1] = (A[2] * A[7] - A[1] * A[8]) * id; Ai[2] = (A[1] * A[5] - A[2] * A[4]) * id;
    Ai[3] = c01 * id; Ai[4] = (A[0] * A[8] - A[2] * A[6]) * id; Ai[5] = (A[2] * A[3] - A[0] * A[5]) * id;
    Ai[6] = c02 * id; Ai[7] = (A[1] * A[6] - A[0] * A[7]) * id; Ai[8] = (A[0] * A[4] - A[1] * A[3]) * id;
    return true;
}

static void putBE(std::vector<uint32_t>& buf, float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    buf.push_back((u >> 24) | ((u >> 8) & 0xFF00u) | ((u << 8) & 0xFF0000u) | (u << 24));
}
static void putBE(std::vector<uint32_t>& buf, int32_t i) {
    uint32_t u = (uint32_t)i;
    buf.push_back((u >> 24) | ((u >> 8) & 0xFF00u) | ((u << 8) & 0xFF0000u) | (u << 24));
}

// ----------------------------------------------------------------------------
// The peridynamic model
// ----------------------------------------------------------------------------
struct Model {
    Params p;
    int N = 0;
    int nz = 0;
    int64_t NB = 0;            // number of (directed) bonds
    double dx, delta, V, K, mu, lam_lame, sc0, dT0max, Wbulk = 0;

    // particle data
    std::vector<double> x0, u, v, f, fo, uzero;  // 3N
    std::vector<double> lamb, m, theta, a, b, T, Wtot;  // N

    // bond data (CSR)
    std::vector<int64_t> off;               // N+1
    std::unique_ptr<int[]> nbr;             // neighbour index
    std::unique_ptr<float[]> xl;            // reference bond length |X|
    std::unique_ptr<float[]> wv;            // influence * partial volume * V_j
    std::unique_ptr<float[]> sc;            // critical stretch
    std::unique_ptr<uint8_t[]> brk;         // 1 = broken

    // post-processing
    std::vector<float> sig, eps;            // 6N each: xx yy zz yz xz xy
    std::vector<float> dmg;                 // N

    // ------------------------------------------------------------------
    void init() {
        dx = p.R / p.nr;
        delta = p.mfac * dx;
        V = dx * dx * dx;
        K = p.E / (3.0 * (1.0 - 2.0 * p.nu));
        mu = p.E / (2.0 * (1.0 + p.nu));
        lam_lame = K - 2.0 * mu / 3.0;
        sc0 = p.sc > 0 ? p.sc : std::sqrt(p.Gc / ((3.0 * mu + std::pow(0.75, 4) * (K - 5.0 * mu / 3.0)) * delta));

        // --- particles on a cubic lattice inside the cylinder ---
        nz = std::max(1, (int)std::lround(p.H / dx));
        const int nxy = 2 * p.nr;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < nxy; ++j)
                for (int i = 0; i < nxy; ++i) {
                    const double x = -p.R + (i + 0.5) * dx, y = -p.R + (j + 0.5) * dx;
                    if (x * x + y * y <= p.R * p.R) {
                        x0.push_back(x); x0.push_back(y); x0.push_back((k + 0.5) * dx);
                    }
                }
        N = (int)(x0.size() / 3);
        u.assign(3 * N, 0.0); v.assign(3 * N, 0.0); f.assign(3 * N, 0.0);
        fo.assign(3 * N, 0.0); uzero.assign(3 * N, 0.0);
        lamb.assign(N, 0.0); m.assign(N, 0.0); theta.assign(N, 0.0);
        a.assign(N, 0.0); b.assign(N, 0.0); T.assign(N, p.Tref); Wtot.assign(N, 0.0);
    }

    // ------------------------------------------------------------------
    // KD-tree neighbour search -> CSR bond list (done once)
    // ------------------------------------------------------------------
    void buildBonds() {
        const double t0 = omp_get_wtime();
        KDTree tree(x0.data(), N);
        const double t1 = omp_get_wtime();
        const double rs = delta + 0.5 * dx, rs2 = rs * rs;

        std::vector<int> cnt(N);
#pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < N; ++i) {
            int c = 0;
            tree.radius(&x0[3 * i], rs2, [&](int j, double) { if (j != i) ++c; });
            cnt[i] = c;
        }
        off.assign(N + 1, 0);
        for (int i = 0; i < N; ++i) off[i + 1] = off[i] + cnt[i];
        NB = off[N];

        nbr = uninit<int>(NB); xl = uninit<float>(NB); wv = uninit<float>(NB);
        sc = uninit<float>(NB); brk = uninit<uint8_t>(NB);

        const uint64_t seed = (uint64_t)p.seed;
        const double wb = p.weibull;
        const double gnorm = wb > 0 ? std::tgamma(1.0 + 1.0 / wb) : 1.0;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            int64_t q = off[i];
            tree.radius(&x0[3 * i], rs2, [&](int j, double) { if (j != i) nbr[q++] = j; });
            std::sort(&nbr[off[i]], &nbr[off[i + 1]]);   // memory locality for gathers
            double mi = 0.0, wt = 0.0;
            for (int64_t q2 = off[i]; q2 < off[i + 1]; ++q2) {
                const int j = nbr[q2];
                const double X0 = x0[3 * j] - x0[3 * i], X1 = x0[3 * j + 1] - x0[3 * i + 1], X2 = x0[3 * j + 2] - x0[3 * i + 2];
                const double L = std::sqrt(X0 * X0 + X1 * X1 + X2 * X2);
                // partial-volume correction
                double beta = 1.0;
                if (L > delta - 0.5 * dx) beta = std::clamp((delta + 0.5 * dx - L) / dx, 0.0, 1.0);
                xl[q2] = (float)L;
                wv[q2] = (float)(beta * V);          // influence function omega = 1
                brk[q2] = 0;
                // symmetric per-bond critical stretch with Weibull scatter
                double s = sc0;
                if (wb > 0) {
                    const uint64_t lo = (uint64_t)std::min(i, j), hi = (uint64_t)std::max(i, j);
                    const uint64_t h = splitmix64((lo << 32 | hi) ^ splitmix64(seed));
                    const double U = ((h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
                    s = sc0 * std::pow(-std::log(1.0 - U), 1.0 / wb) / gnorm;
                }
                sc[q2] = (float)s;
                const double w = (double)wv[q2], Lf = (double)xl[q2];
                mi += w * Lf * Lf;
                wt += w;
            }
            m[i] = mi;
            Wtot[i] = wt;
            b[i] = 15.0 * mu / mi;
        }

        for (int i = 0; i < N; ++i) Wbulk = std::max(Wbulk, Wtot[i]);   // full interior family

        // ADR fictitious mass: Gershgorin-type estimate of the LPS stiffness row sum
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
                st += (double)wv[q] * (b[i] + b[j] +
                      cdil * xl[q] * (Wx[i] / (m[i] * m[i]) + Wx[j] / (m[j] * m[j])));
            }
            lamb[i] = 0.25 * p.sf * st;
        }
        const double t2 = omp_get_wtime();
        std::printf("  KD-tree build %.3f s, radius search + bond setup %.3f s\n", t1 - t0, t2 - t1);
    }

    // ------------------------------------------------------------------
    // temperature field for load fraction 'frac'
    // ------------------------------------------------------------------
    void setTemperature(double frac, double& Tc, double& Ts) {
        const double q = frac * p.qmax;
        const double dT0 = q / (4.0 * M_PI * p.kth);
        Ts = p.Tref + frac * (p.Tsurf - p.Tref);
        Tc = Ts + dT0;
        dT0max = dT0;
        const double R2 = p.R * p.R;
        const double TsL = Ts;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const double r2 = x0[3 * i] * x0[3 * i] + x0[3 * i + 1] * x0[3 * i + 1];
            T[i] = TsL + dT0 * (1.0 - r2 / R2);
        }
    }

    // ------------------------------------------------------------------
    // LPS internal force density.  Pass 1: dilatation + bond breaking.
    // Pass 2: force gather.  Returns number of newly broken (directed) bonds.
    // ------------------------------------------------------------------
    int64_t forces(const double* __restrict U, double* __restrict F, bool allowBreak) {
        const double alpha = p.alpha, Tref = p.Tref;
        const double cA = 3.0 * K - 5.0 * mu;
        const double* __restrict X = x0.data();
        const double* __restrict Tp = T.data();
        const int* __restrict NB_ = nbr.get();
        const float* __restrict XL = xl.get();
        const float* __restrict WV = wv.get();
        const float* __restrict SC = sc.get();
        uint8_t* __restrict BR = brk.get();
        const int64_t* __restrict OF = off.data();
        double* __restrict TH = theta.data();
        double* __restrict A = a.data();
        const double* __restrict Bc = b.data();
        const double* __restrict M = m.data();
        int64_t nb = 0;

#pragma omp parallel
        {
#pragma omp for schedule(static) reduction(+ : nb)
            for (int i = 0; i < N; ++i) {
                const double yi0 = X[3 * i] + U[3 * i], yi1 = X[3 * i + 1] + U[3 * i + 1], yi2 = X[3 * i + 2] + U[3 * i + 2];
                const double Ti = Tp[i];
                double th = 0.0;
                for (int64_t q = OF[i]; q < OF[i + 1]; ++q) {
                    if (BR[q]) continue;
                    const int j = NB_[q];
                    const double d0 = X[3 * j] + U[3 * j] - yi0;
                    const double d1 = X[3 * j + 1] + U[3 * j + 1] - yi1;
                    const double d2 = X[3 * j + 2] + U[3 * j + 2] - yi2;
                    const double y = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
                    const double x = XL[q];
                    const double e = y - x * (1.0 + alpha * (0.5 * (Ti + Tp[j]) - Tref));
                    if (allowBreak && e > (double)SC[q] * x) { BR[q] = 1; ++nb; continue; }
                    th += (double)WV[q] * x * e;
                }
                const double t = 3.0 * th / M[i];
                TH[i] = t;
                A[i] = cA * t / M[i];   // t_ij = A_i |X| + B_i e   (LPS, omega = 1)
            }
            // implicit barrier: all dilatations known
#pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                const double yi0 = X[3 * i] + U[3 * i], yi1 = X[3 * i + 1] + U[3 * i + 1], yi2 = X[3 * i + 2] + U[3 * i + 2];
                const double Ti = Tp[i], Ai = A[i], Bi = Bc[i];
                double f0 = 0, f1 = 0, f2 = 0;
                for (int64_t q = OF[i]; q < OF[i + 1]; ++q) {
                    if (BR[q]) continue;
                    const int j = NB_[q];
                    const double d0 = X[3 * j] + U[3 * j] - yi0;
                    const double d1 = X[3 * j + 1] + U[3 * j + 1] - yi1;
                    const double d2 = X[3 * j + 2] + U[3 * j + 2] - yi2;
                    const double y = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
                    const double x = XL[q];
                    const double e = y - x * (1.0 + alpha * (0.5 * (Ti + Tp[j]) - Tref));
                    const double tsum = (Ai + A[j]) * x + (Bi + Bc[j]) * e;   // t_ij + t_ji
                    const double s = tsum * (double)WV[q] / y;
                    f0 += s * d0; f1 += s * d1; f2 += s * d2;
                }
                F[3 * i] = f0; F[3 * i + 1] = f1; F[3 * i + 2] = f2;
            }
        }
        return nb;
    }

    // max over particles of |f_i| (max-norm: a few freshly broken bonds cannot
    // hide inside a global L2 residual)
    double normInf(const std::vector<double>& w) const {
        double s = 0;
#pragma omp parallel for reduction(max : s) schedule(static)
        for (int i = 0; i < N; ++i)
            s = std::max(s, w[3 * i] * w[3 * i] + w[3 * i + 1] * w[3 * i + 1] + w[3 * i + 2] * w[3 * i + 2]);
        return std::sqrt(s);
    }

    // stretch ratio (mechanical stretch / critical stretch) of an intact bond
    inline double bondRatio(int i, int64_t q) const {
        const int j = nbr[q];
        const double d0 = x0[3 * j] + u[3 * j] - x0[3 * i] - u[3 * i];
        const double d1 = x0[3 * j + 1] + u[3 * j + 1] - x0[3 * i + 1] - u[3 * i + 1];
        const double d2 = x0[3 * j + 2] + u[3 * j + 2] - x0[3 * i + 2] - u[3 * i + 2];
        const double y = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
        const double x = xl[q];
        const double e = y - x * (1.0 + p.alpha * (0.5 * (T[i] + T[j]) - p.Tref));
        return e / ((double)sc[q] * x);
    }

    // Break the most critical bonds: all intact bonds whose stretch ratio is
    // >= max(1, kappa * max_ratio).  kappa -> 1 breaks one "front" at a time
    // (sharp, discrete cracks); kappa = 0 breaks every overstretched bond at once.
    // Decisions use bitwise-identical arithmetic from both ends -> symmetric.
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

    // ------------------------------------------------------------------
    // Adaptive dynamic relaxation for one load step (Kilic & Madenci 2010)
    // ------------------------------------------------------------------
    struct StepInfo { int iters; double res; int64_t broken; };

    // 'scale' = norm of the thermal load (force at zero displacement)
    StepInfo relax(double scale) {
        StepInfo info{0, 1.0, 0};
        const int n3 = 3 * N;
        forces(u.data(), f.data(), false);
        info.res = normInf(f) / scale;
        if (info.res < p.tol) return info;   // already in equilibrium
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n3; ++k) {
            v[k] = 0.5 * f[k] / lamb[k / 3];
            u[k] += v[k];
            fo[k] = f[k];
        }
        for (int it = 1; it <= p.maxit; ++it) {
            forces(u.data(), f.data(), false);
            // adaptive damping coefficient from local diagonal stiffness
            double num = 0, den = 0;
#pragma omp parallel for reduction(+ : num, den) schedule(static)
            for (int k = 0; k < n3; ++k) {
                if (v[k] != 0.0) num -= u[k] * u[k] * (f[k] - fo[k]) / (lamb[k / 3] * v[k]);
                den += u[k] * u[k];
            }
            double c = (den > 0 && num > 0) ? 2.0 * std::sqrt(num / den) : 0.0;
            if (c > 1.999) c = 1.999;
            double vm0 = 0, vm1 = 0, vm2 = 0, fn = 0;
#pragma omp parallel for reduction(+ : vm0, vm1, vm2) reduction(max : fn) schedule(static)
            for (int i = 0; i < N; ++i) {
                const double li = lamb[i];
                for (int d = 0; d < 3; ++d) {
                    const int k = 3 * i + d;
                    v[k] = ((2.0 - c) * v[k] + 2.0 * f[k] / li) / (2.0 + c);
                    fo[k] = f[k];
                }
                fn = std::max(fn, f[3 * i] * f[3 * i] + f[3 * i + 1] * f[3 * i + 1] + f[3 * i + 2] * f[3 * i + 2]);
                vm0 += v[3 * i]; vm1 += v[3 * i + 1]; vm2 += v[3 * i + 2];
            }
            vm0 /= N; vm1 /= N; vm2 /= N;   // remove rigid translation of the free pellet
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                v[3 * i] -= vm0; v[3 * i + 1] -= vm1; v[3 * i + 2] -= vm2;
                u[3 * i] += v[3 * i]; u[3 * i + 1] += v[3 * i + 1]; u[3 * i + 2] += v[3 * i + 2];
            }
            info.iters = it;
            info.res = std::sqrt(fn) / scale;
            if (it > 3 && info.res < p.tol) break;
        }
        return info;
    }

    // ------------------------------------------------------------------
    // Quasi-static load step with sequential bond breaking:
    //   relax to equilibrium -> break the most critical bonds -> relax ...
    // until no bond exceeds s_c (or maxcycles).  Breaking only at equilibrium
    // avoids spurious branching from ADR's fictitious kinetic energy, and
    // breaking only the most critical bonds lets the load redistribute so
    // damage localises into discrete cracks instead of a diffuse band.
    // ------------------------------------------------------------------
    struct LoadInfo { int iters, cycles; double res; int64_t broken; };

    LoadInfo solveLoadStep(bool allowBreak) {
        forces(uzero.data(), f.data(), false);
        const double scale = std::max(normInf(f), 1e-300);
        auto r = relax(scale);
        LoadInfo L{r.iters, 0, r.res, 0};
        if (!allowBreak) return L;
        while (L.cycles < p.maxcycles) {
            double rmax;
            const int64_t nb = breakCritical(rmax);
            if (nb == 0) break;
            L.broken += nb;
            ++L.cycles;
            r = relax(scale);
            L.iters += r.iters;
            L.res = r.res;
            if (p.verbose)
                std::printf("        cycle %5d  max ratio %.3f  broke %6lld  relax iters %5d  resid %.2e\n", L.cycles, rmax,
                            (long long)(nb / 2), r.iters, r.res);
        }
        return L;
    }

    // ------------------------------------------------------------------
    // Stress / strain from the non-local deformation gradient (intact bonds)
    // ------------------------------------------------------------------
    void computeStressStrain() {
        sig.assign(6 * (size_t)N, 0.f); eps.assign(6 * (size_t)N, 0.f); dmg.assign(N, 0.f);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            double Km[9] = {0}, Nm[9] = {0}, Nt[9] = {0}, wint = 0;
            int nint = 0;
            for (int64_t q = off[i]; q < off[i + 1]; ++q) {
                if (brk[q]) continue;
                const int j = nbr[q];
                const double w = wv[q];
                const double ath = p.alpha * (0.5 * (T[i] + T[j]) - p.Tref);   // bond thermal stretch
                double Xb[3], Yb[3];
                for (int d = 0; d < 3; ++d) {
                    Xb[d] = x0[3 * j + d] - x0[3 * i + d];
                    Yb[d] = Xb[d] + u[3 * j + d] - u[3 * i + d];
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
            const float qnan = std::nanf("");
            for (int c = 0; c < 6; ++c) { sig[6 * (size_t)i + c] = qnan; eps[6 * (size_t)i + c] = qnan; }
            // debris, or a crack-side particle whose family is also cut by the free
            // surface: too one-sided for a meaningful F (fit would extrapolate)
            if (dmg[i] > p.dstress || (dmg[i] > 0.05f && wint < p.cover * Wbulk)) continue;
            double Ki[9], det;
            // skip particles whose intact family is too degenerate for a meaningful F
            const double trK = (Km[0] + Km[4] + Km[8]) / 3.0;
            if (nint < 8 || !inv3(Km, Ki, det) || det < 0.02 * trK * trK * trK) continue;
            double Fm[9], Ft[9];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    Fm[3 * r + c] = Nm[3 * r] * Ki[c] + Nm[3 * r + 1] * Ki[3 + c] + Nm[3 * r + 2] * Ki[6 + c];
                    Ft[3 * r + c] = Nt[3 * r] * Ki[c] + Nt[3 * r + 1] * Ki[3 + c] + Nt[3 * r + 2] * Ki[6 + c];
                }
            // total small strain
            const double exx = Fm[0] - 1, eyy = Fm[4] - 1, ezz = Fm[8] - 1;
            const double eyz = 0.5 * (Fm[5] + Fm[7]), exz = 0.5 * (Fm[2] + Fm[6]), exy = 0.5 * (Fm[1] + Fm[3]);
            // elastic strain = total - thermal, with the thermal part fitted over the SAME
            // family using bond-mean temperatures (consistent with the PD mechanical stretch;
            // subtracting alpha*(T_i - Tref) instead biases the result wherever T varies
            // across the horizon - near the surface by several hundred MPa)
            const double ex = exx - Ft[0], ey = eyy - Ft[4], ez = ezz - Ft[8];
            const double gyz = eyz - 0.5 * (Ft[5] + Ft[7]), gxz = exz - 0.5 * (Ft[2] + Ft[6]), gxy = exy - 0.5 * (Ft[1] + Ft[3]);
            const double tr = ex + ey + ez;
            float* s = &sig[6 * (size_t)i];
            float* e = &eps[6 * (size_t)i];
            s[0] = (float)(lam_lame * tr + 2 * mu * ex); s[1] = (float)(lam_lame * tr + 2 * mu * ey);
            s[2] = (float)(lam_lame * tr + 2 * mu * ez); s[3] = (float)(2 * mu * gyz);
            s[4] = (float)(2 * mu * gxz); s[5] = (float)(2 * mu * gxy);
            e[0] = (float)exx; e[1] = (float)eyy; e[2] = (float)ezz; e[3] = (float)eyz; e[4] = (float)exz; e[5] = (float)exy;
        }
    }

    // cylindrical components of a symmetric tensor (xx yy zz yz xz xy) at particle i
    void cyl(const float* t, int i, double& rr, double& tt, double& zz) const {
        const double x = x0[3 * i], y = x0[3 * i + 1];
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

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------
    void writeVTK(const std::string& fn) const {
        FILE* fp = std::fopen(fn.c_str(), "wb");
        if (!fp) { std::fprintf(stderr, "cannot write %s\n", fn.c_str()); return; }
        std::vector<uint32_t> buf;
        buf.reserve(9 * (size_t)N);
        auto flush = [&]() { std::fwrite(buf.data(), 4, buf.size(), fp); buf.clear(); };
        std::fprintf(fp, "# vtk DataFile Version 3.0\nperidynamic fuel pellet\nBINARY\nDATASET POLYDATA\nPOINTS %d float\n", N);
        for (int k = 0; k < 3 * N; ++k) putBE(buf, (float)x0[k]);
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
        scalar("dilatation", [&](int i) { return theta[i]; });
        auto cylS = [&](int i, int c) { double r, t, z; cyl(&sig[6 * (size_t)i], i, r, t, z); return c == 0 ? r : (c == 1 ? t : z); };
        auto cylE = [&](int i, int c) { double r, t, z; cyl(&eps[6 * (size_t)i], i, r, t, z); return c == 0 ? r : (c == 1 ? t : z); };
        scalar("sigma_rr_MPa", [&](int i) { return cylS(i, 0) * 1e-6; });
        scalar("sigma_tt_MPa", [&](int i) { return cylS(i, 1) * 1e-6; });
        scalar("sigma_zz_MPa", [&](int i) { return cylS(i, 2) * 1e-6; });
        scalar("von_mises_MPa", [&](int i) { return vonMises(&sig[6 * (size_t)i]) * 1e-6; });
        scalar("eps_rr", [&](int i) { return cylE(i, 0); });
        scalar("eps_tt", [&](int i) { return cylE(i, 1); });
        scalar("eps_zz", [&](int i) { return cylE(i, 2); });
        std::fprintf(fp, "VECTORS displacement float\n");
        for (int k = 0; k < 3 * N; ++k) putBE(buf, (float)u[k]);
        flush();
        std::fprintf(fp, "\nTENSORS stress_Pa float\n");
        for (int i = 0; i < N; ++i) {
            const float* s = &sig[6 * (size_t)i];
            const float t9[9] = {s[0], s[5], s[4], s[5], s[1], s[3], s[4], s[3], s[2]};
            for (float x : t9) putBE(buf, x);
        }
        flush();
        std::fprintf(fp, "\nTENSORS strain float\n");
        for (int i = 0; i < N; ++i) {
            const float* s = &eps[6 * (size_t)i];
            const float t9[9] = {s[0], s[5], s[4], s[5], s[1], s[3], s[4], s[3], s[2]};
            for (float x : t9) putBE(buf, x);
        }
        flush();
        std::fclose(fp);
    }

    // mid-plane radial profile, PD vs analytic (long solid cylinder, parabolic T,
    // generalised plane strain, free surface):
    //   s_rr = A (r^2/R^2 - 1),  s_tt = A (3 r^2/R^2 - 1),  s_zz = 2A (2 r^2/R^2 - 1)
    //   A = alpha E dT0 / (4 (1 - nu)),  dT0 = T_centre - T_surface
    // returns RMS error of s_tt relative to its surface value (intact particles)
    double writeProfile(const std::string& fn) const {
        const int nb = p.nr;
        const double zmid = 0.5 * nz * dx;
        std::vector<double> sr(nb, 0), st(nb, 0), sz(nb, 0), et(nb, 0), dm(nb, 0), cnt(nb, 0);
        for (int i = 0; i < N; ++i) {
            if (std::fabs(x0[3 * i + 2] - zmid) > 0.75 * dx) continue;
            const double r = std::hypot(x0[3 * i], x0[3 * i + 1]);
            const int k = std::min(nb - 1, (int)(r / p.R * nb));
            double rr, tt, zz;
            cyl(&sig[6 * (size_t)i], i, rr, tt, zz);
            double er, ett, ez;
            cyl(&eps[6 * (size_t)i], i, er, ett, ez);
            if (std::isnan(rr)) continue;
            sr[k] += rr; st[k] += tt; sz[k] += zz; et[k] += ett; dm[k] += dmg[i]; cnt[k] += 1;
        }
        const double A = p.alpha * p.E * dT0max / (4.0 * (1.0 - p.nu));
        FILE* fp = std::fopen(fn.c_str(), "w");
        if (fp) std::fprintf(fp, "r_mm,sigma_rr_MPa,sigma_tt_MPa,sigma_zz_MPa,eps_tt,damage,an_sigma_rr_MPa,an_sigma_tt_MPa,an_sigma_zz_MPa\n");
        double err = 0, nerr = 0;
        for (int k = 0; k < nb; ++k) {
            if (cnt[k] == 0) continue;
            const double r = (k + 0.5) / nb * p.R, xi = r * r / (p.R * p.R);
            const double ar = A * (xi - 1), at = A * (3 * xi - 1), az = 2 * A * (2 * xi - 1);
            const double c = cnt[k];
            if (fp) std::fprintf(fp, "%.5f,%.4f,%.4f,%.4f,%.6e,%.4f,%.4f,%.4f,%.4f\n", r * 1e3,
                                 sr[k] / c * 1e-6, st[k] / c * 1e-6, sz[k] / c * 1e-6, et[k] / c, dm[k] / c,
                                 ar * 1e-6, at * 1e-6, az * 1e-6);
            err += (st[k] / c - at) * (st[k] / c - at); nerr += 1;
        }
        if (fp) std::fclose(fp);
        return (nerr > 0 && A > 0) ? std::sqrt(err / nerr) / (2 * A) : 0.0;
    }

    // count radial cracks: angular sectors near the mid-plane surface with high damage
    int countRadialCracks() const {
        const int nbins = 96;
        std::vector<float> mx(nbins, 0.f);
        const double zmid = 0.5 * nz * dx;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(x0[3 * i + 2] - zmid) > 0.75 * dx) continue;
            const double r = std::hypot(x0[3 * i], x0[3 * i + 1]);
            if (r < 0.75 * p.R) continue;
            double ang = std::atan2(x0[3 * i + 1], x0[3 * i]) + M_PI;
            const int k = std::min(nbins - 1, (int)(ang / (2 * M_PI) * nbins));
            mx[k] = std::max(mx[k], dmg[i]);
        }
        int runs = 0, on = 0;
        for (int k = 0; k < nbins; ++k) {
            const bool c = mx[k] > 0.4f, prev = mx[(k + nbins - 1) % nbins] > 0.4f;
            on += c;
            if (c && !prev) ++runs;
        }
        return (on == nbins) ? -1 : runs;
    }
};

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    Params P;
    if (!parseArgs(argc, argv, P)) { usage(); return 1; }
    mkdir(P.out.c_str(), 0755);

    Model M;
    M.p = P;
    const double tstart = omp_get_wtime();
    std::printf("=== Peridynamic fuel pellet thermal cracking (LPS, 3D) ===\n");
    std::printf("threads: %d\n", omp_get_max_threads());
    M.init();
    std::printf("pellet R = %.3f mm, H = %.3f mm | dx = %.4f mm, delta = %.4f mm | particles = %d\n",
                P.R * 1e3, M.nz * M.dx * 1e3, M.dx * 1e3, M.delta * 1e3, M.N);
    std::printf("E = %.0f GPa, nu = %.3f, K = %.1f GPa, mu = %.1f GPa, alpha = %.2e /K, Gc = %.2f J/m2\n",
                P.E * 1e-9, P.nu, M.K * 1e-9, M.mu * 1e-9, P.alpha, P.Gc);
    std::printf("critical stretch s_c0 = %.4e  (Weibull m = %g)%s\n", M.sc0, P.weibull,
                P.elastic ? "  [fracture DISABLED]" : "");

    M.buildBonds();
    std::printf("bonds (directed) = %lld, avg family size = %.1f, memory ~ %.1f MB\n",
                (long long)M.NB, (double)M.NB / M.N, M.NB * 17.0 / 1048576.0 + M.N * 200.0 / 1048576.0);

    FILE* hist = std::fopen((P.out + "/history.csv").c_str(), "w");
    if (hist) std::fprintf(hist, "step,q_kW_per_m,T_centre_K,T_surface_K,adr_iters,residual,broken_bonds,mean_damage,max_sigma_tt_MPa,radial_cracks\n");

    std::printf("\n%5s %8s %8s %8s %7s %6s %9s %11s %8s %10s %7s %7s\n", "step", "q'kW/m", "Tc[K]", "Ts[K]", "iters",
                "cycles", "resid", "broken", "meanD", "maxStt", "cracks", "t[s]");
    int64_t brokenTotal = 0;
    double firstCrackQ = -1;
    double tForce = 0;
    long long itTotal = 0;
    for (int step = 1; step <= P.steps; ++step) {
        const double frac = (double)step / P.steps;
        double Tc, Ts;
        M.setTemperature(frac, Tc, Ts);
        const double t0 = omp_get_wtime();
        auto info = M.solveLoadStep(!P.elastic);
        const double t1 = omp_get_wtime();
        tForce += t1 - t0;
        itTotal += info.iters + 2 * (info.cycles + 1);
        brokenTotal += info.broken;
        if (firstCrackQ < 0 && brokenTotal > 0) firstCrackQ = frac * P.qmax;

        M.computeStressStrain();
        double meanD = 0, maxStt = -1e300;
        for (int i = 0; i < M.N; ++i) {
            meanD += M.dmg[i];
            double r, t, z;
            M.cyl(&M.sig[6 * (size_t)i], i, r, t, z);
            if (!std::isnan(t)) maxStt = std::max(maxStt, t);
        }
        meanD /= M.N;
        const int cracks = M.countRadialCracks();
        std::printf("%5d %8.2f %8.1f %8.1f %7d %6d %9.2e %11lld %8.4f %10.1f %7d %7.1f\n", step, frac * P.qmax * 1e-3, Tc, Ts,
                    info.iters, info.cycles, info.res, (long long)(brokenTotal / 2), meanD, maxStt * 1e-6, cracks, t1 - t0);
        std::fflush(stdout);
        if (hist) {
            std::fprintf(hist, "%d,%.4f,%.2f,%.2f,%d,%.4e,%lld,%.6f,%.3f,%d\n", step, frac * P.qmax * 1e-3, Tc, Ts,
                         info.iters, info.res, (long long)(brokenTotal / 2), meanD, maxStt * 1e-6, cracks);
            std::fflush(hist);
        }
        if (step % P.every == 0 || step == P.steps) {
            char name[64];
            std::snprintf(name, sizeof(name), "/profile_%04d.csv", step);
            const double err = M.writeProfile(P.out + name);
            if (brokenTotal == 0)
                std::printf("      mid-plane hoop stress vs analytic: RMS error = %.1f %% of surface value\n", 100 * err);
            if (!P.novtk) {
                std::snprintf(name, sizeof(name), "/pellet_%04d.vtk", step);
                M.writeVTK(P.out + name);
            }
        }
    }
    if (hist) std::fclose(hist);

    const double ttot = omp_get_wtime() - tstart;
    std::printf("\nsummary\n");
    if (firstCrackQ > 0) std::printf("  first bond failure at q' <= %.2f kW/m\n", firstCrackQ * 1e-3);
    std::printf("  broken bonds: %lld of %lld (%.2f %%)\n", (long long)(brokenTotal / 2), (long long)(M.NB / 2),
                100.0 * brokenTotal / std::max<int64_t>(1, M.NB));
    std::printf("  solver time %.2f s, total %.2f s, throughput %.1f M bond-evals/s\n", tForce, ttot,
                2.0 * M.NB * itTotal / tForce * 1e-6);
    std::printf("  output: %s/\n", P.out.c_str());
    return 0;
}
