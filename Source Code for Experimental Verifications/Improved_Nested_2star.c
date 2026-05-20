/*
 * Large but simple toy experiments for the Nested-2* LWE collision framework.
 *
 * The notation follows the paper:
 *
 *      s  = s1 + s2 + s3 + s4,
 *      a1 = s1 + s2,
 *      a2 = s3 + s4,
 *      s_i in T_i,
 *      a1 in D1, a2 in D2, s in D.
 *
 * The program validates, on large toy parameters where exact enumeration is
 * still possible, the random-projection and representation-count heuristics
 * used in the nested-collision analysis. It is not a cryptographic-size attack
 * implementation and it does not perform numerical optimization.
 *
 * 
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -------------------------- toy parameters -------------------------- */

#define BLOCK_LEN       4
#define JOINT_LEN       8
#define NUM_BLOCKS      4
#define N               (NUM_BLOCKS * BLOCK_LEN + JOINT_LEN)

/* Layer-1 projection: Z_q^ell.  q^ell = 149^2 = 22201. */
#define Q               149
#define ELL             2
#define ROWS            (2 * ELL)

/* The default trial count is intentionally large. */
#define DEFAULT_TRIALS         10000LL
#define PROGRESS_EVERY_TRIALS  100LL
#define HIT_REPEATS_PER_TRIAL  5

/* ----------------------------- types -------------------------------- */

typedef int8_t Vec[N];
typedef int8_t BlockVec[BLOCK_LEN];
typedef int8_t JointVec[JOINT_LEN];

typedef struct {
    const char *name;
    int size;
    Vec *v;
} Domain;

typedef struct {
    int size;
    int capacity;
    JointVec *v;
} JointList;

typedef struct {
    int size;
    int capacity;
    BlockVec *v;
} BlockList;

/* ----------------------- global experiment state --------------------- */

static int A[ROWS][N];
static uint64_t rng_state = 0x6d2b79f5aa1234efULL;

/* Histograms for layer-1 collision counts. */
static int *hist_T1 = NULL;
static int *hist_T2_random = NULL;
static int *hist_T2_correct = NULL;
static int *hist_rand1 = NULL;
static int *hist_rand2 = NULL;

/* Linked lists for uniform sampling from the actual correct-r collision set. */
static int *head_T1 = NULL;
static int *head_T2_correct = NULL;
static int *next_T1 = NULL;
static int *next_T2_correct = NULL;

/* Temporary array for exact computation of E_r. */
static unsigned char *seen = NULL;

/* Collision-bucket cumulative distribution. */
static int *bucket_id = NULL;
static long long *bucket_cum = NULL;
static int bucket_count = 0;

/* ---------------------------- utilities ----------------------------- */

static void die(const char *msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

static uint64_t splitmix64(void) {
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static int rnd_int(int m) {
    return (int)(splitmix64() % (uint64_t)m);
}

static long long rnd_ll(long long m) {
    return (long long)(splitmix64() % (uint64_t)m);
}

static long long ipow_ll(int base, int exp) {
    long long r = 1;
    for (int i = 0; i < exp; i++) r *= base;
    return r;
}

static int modq(int x) {
    int r = x % Q;
    return (r < 0) ? r + Q : r;
}

static int vec_equal(const Vec a, const Vec b) {
    return memcmp(a, b, sizeof(Vec)) == 0;
}

static void vec_zero(Vec x) {
    memset(x, 0, sizeof(Vec));
}

static void vec_add(Vec out, const Vec a, const Vec b) {
    for (int i = 0; i < N; i++) out[i] = (int8_t)((int)a[i] + (int)b[i]);
}

static void vec_sub(Vec out, const Vec a, const Vec b) {
    for (int i = 0; i < N; i++) out[i] = (int8_t)((int)a[i] - (int)b[i]);
}

static void copy_block(Vec x, int block, const BlockVec b) {
    int off = block * BLOCK_LEN;
    for (int i = 0; i < BLOCK_LEN; i++) x[off + i] = b[i];
}

static void copy_joint(Vec x, const JointVec j) {
    int off = NUM_BLOCKS * BLOCK_LEN;
    for (int i = 0; i < JOINT_LEN; i++) x[off + i] = j[i];
}

static void print_vec(const char *label, const Vec x) {
    printf("%s = [", label);
    for (int i = 0; i < N; i++) {
        printf("%d", (int)x[i]);
        if (i + 1 < N) printf(",");
    }
    printf("]\n");
}

/* ----------------------- pattern generation ------------------------- */

static void blocklist_init(BlockList *L, int capacity) {
    L->size = 0;
    L->capacity = capacity;
    L->v = (BlockVec *)calloc((size_t)capacity, sizeof(BlockVec));
    if (!L->v) die("could not allocate block pattern list");
}

static void jointlist_init(JointList *L, int capacity) {
    L->size = 0;
    L->capacity = capacity;
    L->v = (JointVec *)calloc((size_t)capacity, sizeof(JointVec));
    if (!L->v) die("could not allocate joint pattern list");
}

static void blocklist_push(BlockList *L, const BlockVec b) {
    if (L->size >= L->capacity) die("block pattern capacity too small");
    memcpy(L->v[L->size++], b, sizeof(BlockVec));
}

static void jointlist_push(JointList *L, const JointVec j) {
    if (L->size >= L->capacity) die("joint pattern capacity too small");
    memcpy(L->v[L->size++], j, sizeof(JointVec));
}

/* tau^{BLOCK_LEN}(1): one +1 and one -1 in the block. */
static void generate_block_tau(BlockList *B) {
    blocklist_init(B, BLOCK_LEN * (BLOCK_LEN - 1));
    for (int p = 0; p < BLOCK_LEN; p++) {
        for (int m = 0; m < BLOCK_LEN; m++) {
            if (m == p) continue;
            BlockVec b = {0};
            b[p] = 1;
            b[m] = -1;
            blocklist_push(B, b);
        }
    }
}

/* tau_2^{JOINT_LEN}(1,1): one each of +1,-1,+2,-2. */
static void generate_joint_tau2_lower(JointList *L) {
    jointlist_init(L, JOINT_LEN * (JOINT_LEN - 1) * (JOINT_LEN - 2) * (JOINT_LEN - 3));
    for (int p1 = 0; p1 < JOINT_LEN; p1++) {
        for (int m1 = 0; m1 < JOINT_LEN; m1++) {
            if (m1 == p1) continue;
            for (int p2 = 0; p2 < JOINT_LEN; p2++) {
                if (p2 == p1 || p2 == m1) continue;
                for (int m2 = 0; m2 < JOINT_LEN; m2++) {
                    if (m2 == p1 || m2 == m1 || m2 == p2) continue;
                    JointVec j = {0};
                    j[p1] = 1;
                    j[m1] = -1;
                    j[p2] = 2;
                    j[m2] = -2;
                    jointlist_push(L, j);
                }
            }
        }
    }
}

static void gen_joint_mid_rec(JointList *L, int pos,
                              int p1, int m1, int p2, int m2, JointVec cur) {
    if (pos == JOINT_LEN) {
        if (p1 == 0 && m1 == 0 && p2 == 0 && m2 == 0) jointlist_push(L, cur);
        return;
    }

    int remaining = JOINT_LEN - pos - 1;

    cur[pos] = 0;
    if (p1 + m1 + p2 + m2 <= remaining)
        gen_joint_mid_rec(L, pos + 1, p1, m1, p2, m2, cur);

    if (p1 > 0 && (p1 - 1) + m1 + p2 + m2 <= remaining) {
        cur[pos] = 1;
        gen_joint_mid_rec(L, pos + 1, p1 - 1, m1, p2, m2, cur);
    }
    if (m1 > 0 && p1 + (m1 - 1) + p2 + m2 <= remaining) {
        cur[pos] = -1;
        gen_joint_mid_rec(L, pos + 1, p1, m1 - 1, p2, m2, cur);
    }
    if (p2 > 0 && p1 + m1 + (p2 - 1) + m2 <= remaining) {
        cur[pos] = 2;
        gen_joint_mid_rec(L, pos + 1, p1, m1, p2 - 1, m2, cur);
    }
    if (m2 > 0 && p1 + m1 + p2 + (m2 - 1) <= remaining) {
        cur[pos] = -2;
        gen_joint_mid_rec(L, pos + 1, p1, m1, p2, m2 - 1, cur);
    }
    cur[pos] = 0;
}

/* Middle joint domain: tau_2^{JOINT_LEN}(2,1). */
static void generate_joint_tau2_middle(JointList *L) {
    jointlist_init(L, 6000);
    JointVec cur = {0};
    gen_joint_mid_rec(L, 0, 2, 2, 1, 1, cur);
}

static void gen_joint_secret_rec(JointList *L, int pos, int p1, int m1, JointVec cur) {
    if (pos == JOINT_LEN) {
        if (p1 == 0 && m1 == 0) jointlist_push(L, cur);
        return;
    }

    int remaining = JOINT_LEN - pos - 1;

    cur[pos] = 0;
    if (p1 + m1 <= remaining)
        gen_joint_secret_rec(L, pos + 1, p1, m1, cur);

    if (p1 > 0 && (p1 - 1) + m1 <= remaining) {
        cur[pos] = 1;
        gen_joint_secret_rec(L, pos + 1, p1 - 1, m1, cur);
    }

    if (m1 > 0 && p1 + (m1 - 1) <= remaining) {
        cur[pos] = -1;
        gen_joint_secret_rec(L, pos + 1, p1, m1 - 1, cur);
    }

    cur[pos] = 0;
}

/* Secret joint domain: tau^{JOINT_LEN}(2), i.e. two +1 and two -1. */
static void generate_joint_secret(JointList *L) {
    jointlist_init(L, 500);
    JointVec cur = {0};
    gen_joint_secret_rec(L, 0, 2, 2, cur);
}

/* ------------------------- domain generation ------------------------ */

static void domain_alloc(Domain *D, const char *name, int size) {
    D->name = name;
    D->size = size;
    D->v = (Vec *)calloc((size_t)size, sizeof(Vec));
    if (!D->v) die("could not allocate domain");
}

static void build_Ti(Domain T[NUM_BLOCKS], const BlockList *B, const JointList *Jlower) {
    int size = B->size * Jlower->size;
    for (int which = 0; which < NUM_BLOCKS; which++) {
        char *name = (char *)malloc(16);
        if (!name) die("could not allocate domain name");
        snprintf(name, 16, "T%d", which + 1);
        domain_alloc(&T[which], name, size);
        int idx = 0;
        for (int b = 0; b < B->size; b++) {
            for (int j = 0; j < Jlower->size; j++) {
                vec_zero(T[which].v[idx]);
                copy_block(T[which].v[idx], which, B->v[b]);
                copy_joint(T[which].v[idx], Jlower->v[j]);
                idx++;
            }
        }
    }
}

static void build_D1_D2(Domain *D1, Domain *D2, const BlockList *B, const JointList *Jmiddle) {
    int size = B->size * B->size * Jmiddle->size;
    domain_alloc(D1, "D1", size);
    domain_alloc(D2, "D2", size);

    int idx1 = 0, idx2 = 0;
    for (int bA = 0; bA < B->size; bA++) {
        for (int bB = 0; bB < B->size; bB++) {
            for (int j = 0; j < Jmiddle->size; j++) {
                vec_zero(D1->v[idx1]);
                copy_block(D1->v[idx1], 0, B->v[bA]);
                copy_block(D1->v[idx1], 1, B->v[bB]);
                copy_joint(D1->v[idx1], Jmiddle->v[j]);
                idx1++;

                vec_zero(D2->v[idx2]);
                copy_block(D2->v[idx2], 2, B->v[bA]);
                copy_block(D2->v[idx2], 3, B->v[bB]);
                copy_joint(D2->v[idx2], Jmiddle->v[j]);
                idx2++;
            }
        }
    }
}

static void build_D(Domain *D, const BlockList *B, const JointList *Jsecret) {
    int size = B->size * B->size * B->size * B->size * Jsecret->size;
    domain_alloc(D, "D", size);

    int idx = 0;
    for (int b0 = 0; b0 < B->size; b0++) {
        for (int b1 = 0; b1 < B->size; b1++) {
            for (int b2 = 0; b2 < B->size; b2++) {
                for (int b3 = 0; b3 < B->size; b3++) {
                    for (int j = 0; j < Jsecret->size; j++) {
                        vec_zero(D->v[idx]);
                        copy_block(D->v[idx], 0, B->v[b0]);
                        copy_block(D->v[idx], 1, B->v[b1]);
                        copy_block(D->v[idx], 2, B->v[b2]);
                        copy_block(D->v[idx], 3, B->v[b3]);
                        copy_joint(D->v[idx], Jsecret->v[j]);
                        idx++;
                    }
                }
            }
        }
    }
}

/* --------------------- planted toy representation ------------------- */

static void build_planted_vectors(const BlockList *B,
                                  Vec s1, Vec s2, Vec s3, Vec s4,
                                  Vec a1, Vec a2, Vec s) {
    /* These joint vectors are chosen so that
     *   s1+s2 = a1 in tau_2^8(2,1),
     *   s3+s4 = a2 in tau_2^8(2,1),
     *   a1+a2 = s  in tau^8(2).
     */
    const JointVec j_s1 = { 1,-1, 2,-2, 0, 0, 0, 0};
    const JointVec j_s2 = { 0, 0,-1, 1, 2,-2, 0, 0};
    const JointVec j_s3 = { 1,-1,-2, 2, 0, 0, 0, 0};
    const JointVec j_s4 = {-2, 2, 0, 0,-1, 1, 0, 0};

    vec_zero(s1); vec_zero(s2); vec_zero(s3); vec_zero(s4);
    copy_block(s1, 0, B->v[0]); copy_joint(s1, j_s1);
    copy_block(s2, 1, B->v[0]); copy_joint(s2, j_s2);
    copy_block(s3, 2, B->v[0]); copy_joint(s3, j_s3);
    copy_block(s4, 3, B->v[0]); copy_joint(s4, j_s4);

    Vec tmp;
    vec_add(a1, s1, s2);
    vec_add(a2, s3, s4);
    vec_add(tmp, a1, a2);
    memcpy(s, tmp, sizeof(Vec));
}

/* ---------------------------- hash table ---------------------------- */

typedef struct {
    const Domain *D;
    int capacity;
    int *slot;        /* -1 means empty; otherwise stores domain index. */
} VecHash;

static uint64_t vec_hash(const Vec x) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < N; i++) {
        uint8_t b = (uint8_t)((int)x[i] + 16);
        h ^= (uint64_t)b;
        h *= 1099511628211ULL;
    }
    return h;
}

static int next_power_of_two(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

static void vechash_build(VecHash *H, const Domain *D) {
    H->D = D;
    H->capacity = next_power_of_two(4 * D->size + 16);
    H->slot = (int *)malloc((size_t)H->capacity * sizeof(int));
    if (!H->slot) die("could not allocate vector hash table");
    for (int i = 0; i < H->capacity; i++) H->slot[i] = -1;

    for (int idx = 0; idx < D->size; idx++) {
        uint64_t h = vec_hash(D->v[idx]);
        int pos = (int)(h & (uint64_t)(H->capacity - 1));
        while (H->slot[pos] != -1) {
            if (vec_equal(D->v[H->slot[pos]], D->v[idx])) {
                die("duplicate vector detected in a domain");
            }
            pos = (pos + 1) & (H->capacity - 1);
        }
        H->slot[pos] = idx;
    }
}

static int vechash_find(const VecHash *H, const Vec x) {
    uint64_t h = vec_hash(x);
    int pos = (int)(h & (uint64_t)(H->capacity - 1));
    while (H->slot[pos] != -1) {
        int idx = H->slot[pos];
        if (vec_equal(H->D->v[idx], x)) return idx;
        pos = (pos + 1) & (H->capacity - 1);
    }
    return -1;
}

static void vechash_free(VecHash *H) {
    free(H->slot);
    H->slot = NULL;
    H->capacity = 0;
    H->D = NULL;
}

/* Count # {(x,y) in L x R : x+y=target}.  Optionally store left indices. */
static int count_representations(const Domain *L, const Domain *R, const Vec target, int **left_indices_out) {
    VecHash HR;
    vechash_build(&HR, R);

    int *left = NULL;
    if (left_indices_out) {
        left = (int *)malloc((size_t)L->size * sizeof(int));
        if (!left) die("could not allocate representation list");
    }

    int count = 0;
    Vec need;
    for (int i = 0; i < L->size; i++) {
        vec_sub(need, target, L->v[i]);
        if (vechash_find(&HR, need) >= 0) {
            if (left) left[count] = i;
            count++;
        }
    }

    vechash_free(&HR);
    if (left_indices_out) *left_indices_out = left;
    return count;
}

/* ----------------------------- projections -------------------------- */

static void random_A(void) {
    for (int r = 0; r < ROWS; r++) {
        for (int j = 0; j < N; j++) A[r][j] = rnd_int(Q);
    }
}

static void projection_digits(const Vec x, int rows, int out[]) {
    for (int r = 0; r < rows; r++) {
        int s = 0;
        for (int j = 0; j < N; j++) s += A[r][j] * (int)x[j];
        out[r] = modq(s);
    }
}

static int encode_digits(const int digits[], int rows) {
    int idx = 0;
    int base = 1;
    for (int r = 0; r < rows; r++) {
        idx += digits[r] * base;
        base *= Q;
    }
    return idx;
}

static int encode_difference(const int a[], const int b[], int rows) {
    int idx = 0;
    int base = 1;
    for (int r = 0; r < rows; r++) {
        idx += modq(a[r] - b[r]) * base;
        base *= Q;
    }
    return idx;
}

static long long collision_count(const int *h1, const int *h2, int range) {
    long long c = 0;
    for (int i = 0; i < range; i++) c += (long long)h1[i] * (long long)h2[i];
    return c;
}

/* ---------------------- collision-pair sampling --------------------- */

static int kth_from_head(int head, const int *next, int k) {
    int idx = head;
    for (int i = 0; i < k; i++) idx = next[idx];
    return idx;
}

static int is_distinguished_pair(const Domain *L, const Domain *R,
                                 int xi, int yi, const Vec target) {
    for (int j = 0; j < N; j++) {
        if ((int)L->v[xi][j] + (int)R->v[yi][j] != (int)target[j]) return 0;
    }
    return 1;
}

static void build_collision_bucket_distribution(int range) {
    bucket_count = 0;
    long long total = 0;
    for (int b = 0; b < range; b++) {
        long long w = (long long)hist_T1[b] * (long long)hist_T2_correct[b];
        if (w > 0) {
            total += w;
            bucket_id[bucket_count] = b;
            bucket_cum[bucket_count] = total;
            bucket_count++;
        }
    }
}

static int sample_collision_bucket(long long total_collisions) {
    long long u = rnd_ll(total_collisions);
    int lo = 0, hi = bucket_count - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (u < bucket_cum[mid]) hi = mid;
        else lo = mid + 1;
    }
    return bucket_id[lo];
}

static int sample_collision_is_distinguished(const Domain *L, const Domain *R,
                                             long long total_collisions,
                                             const Vec target) {
    int b = sample_collision_bucket(total_collisions);
    int x_pos = rnd_int(hist_T1[b]);
    int y_pos = rnd_int(hist_T2_correct[b]);
    int xi = kth_from_head(head_T1[b], next_T1, x_pos);
    int yi = kth_from_head(head_T2_correct[b], next_T2_correct, y_pos);
    return is_distinguished_pair(L, R, xi, yi, target);
}

/* ------------------------------- main ------------------------------- */

int main(int argc, char **argv) {
    long long total_trials = DEFAULT_TRIALS;
    if (argc >= 2) {
        char *endptr = NULL;
        total_trials = strtoll(argv[1], &endptr, 10);
        if (endptr == argv[1] || *endptr != '\0' || total_trials <= 0) {
            die("trial-count argument must be a positive integer");
        }
    }

    const int range = (int)ipow_ll(Q, ELL);
    const long long range2 = ipow_ll(Q, 2 * ELL);

    BlockList B;
    JointList Jlower, Jmiddle, Jsecret;
    generate_block_tau(&B);
    generate_joint_tau2_lower(&Jlower);
    generate_joint_tau2_middle(&Jmiddle);
    generate_joint_secret(&Jsecret);

    Domain mD, mD1, mD2, mT[NUM_BLOCKS];
    build_Ti(mT, &B, &Jlower);
    build_D1_D2(&mD1, &mD2, &B, &Jmiddle);
    build_D(&mD, &B, &Jsecret);

    Vec s1, s2, s3, s4, a1, a2, s;
    build_planted_vectors(&B, s1, s2, s3, s4, a1, a2, s);

    int *R2_a1_left = NULL;
    int *R2_a2_left = NULL;
    int *R1_s_left = NULL;
    const int R2_a1 = count_representations(&mT[0], &mT[1], a1, &R2_a1_left);
    const int R2_a2 = count_representations(&mT[2], &mT[3], a2, &R2_a2_left);
    const int R1_s  = count_representations(&mD1, &mD2, s, &R1_s_left);

    hist_T1 = (int *)calloc((size_t)range, sizeof(int));
    hist_T2_random = (int *)calloc((size_t)range, sizeof(int));
    hist_T2_correct = (int *)calloc((size_t)range, sizeof(int));
    hist_rand1 = (int *)calloc((size_t)range, sizeof(int));
    hist_rand2 = (int *)calloc((size_t)range, sizeof(int));
    head_T1 = (int *)malloc((size_t)range * sizeof(int));
    head_T2_correct = (int *)malloc((size_t)range * sizeof(int));
    next_T1 = (int *)malloc((size_t)mT[0].size * sizeof(int));
    next_T2_correct = (int *)malloc((size_t)mT[1].size * sizeof(int));
    seen = (unsigned char *)calloc((size_t)range, sizeof(unsigned char));
    bucket_id = (int *)malloc((size_t)range * sizeof(int));
    bucket_cum = (long long *)malloc((size_t)range * sizeof(long long));
    if (!hist_T1 || !hist_T2_random || !hist_T2_correct || !hist_rand1 || !hist_rand2 ||
        !head_T1 || !head_T2_correct || !next_T1 || !next_T2_correct ||
        !seen || !bucket_id || !bucket_cum) {
        die("could not allocate experiment arrays");
    }

    printf("Large toy validation for Nested-2* LWE heuristics\n");
    printf("Trials: %lld\n", total_trials);
    printf("Parameters: n=%d = 4*%d + %d, q=%d, ell=%d, q^ell=%d, q^(2ell)=%lld\n",
           N, BLOCK_LEN, JOINT_LEN, Q, ELL, range, range2);
    printf("Domains:\n");
    printf("  |D|        = %d\n", mD.size);
    printf("  |T_i|      = %d for each i=1,2,3,4\n", mT[0].size);
    printf("  |D1|=|D2|  = %d\n", mD1.size);
    printf("Effective lambda from q^ell = |D|^(1/2+lambda): %.6f\n",
           log((double)range) / log((double)mD.size) - 0.5);
    printf("Balancing check q^ell/|T_i| = %.6f\n", (double)range / (double)mT[0].size);
    printf("Exact representation counts:\n");
    printf("  R2(a1) = # {(s1,s2) in T1 x T2 : s1+s2=a1} = %d\n", R2_a1);
    printf("  R2(a2) = # {(s3,s4) in T3 x T4 : s3+s4=a2} = %d\n", R2_a2);
    printf("  R1(s)  = # {(a1,a2) in D1 x D2 : a1+a2=s}  = %d\n", R1_s);
    printf("Constraint checks: R1 < q^ell is %s; R2(a1)^2 < q^ell is %s; R2(a2)^2 < q^ell is %s.\n",
           (R1_s < range ? "true" : "false"),
           ((long long)R2_a1 * R2_a1 < range ? "true" : "false"),
           ((long long)R2_a2 * R2_a2 < range ? "true" : "false"));
    print_vec("s", s);
    print_vec("a1", a1);
    print_vec("a2", a2);

    double false_sum = 0.0;
    double unique_sum = 0.0;
    double lower_lwe_sum = 0.0, lower_lwe_sq = 0.0;
    double lower_rand_sum = 0.0, lower_rand_sq = 0.0;
    double lower_correct_sum = 0.0;
    double Er_middle_sum = 0.0;
    double hit_sum = 0.0, hit_sq = 0.0;
    long long hit_trials = 0;
    long long trials = 0;

    while (trials < total_trials) {
        random_A();

        int s_digits[ROWS];
        int a1_digits[ELL];
        projection_digits(s, ROWS, s_digits);
        projection_digits(a1, ELL, a1_digits);

        int r_random[ELL];
        for (int i = 0; i < ELL; i++) r_random[i] = rnd_int(Q);

        /* [1] Projected uniqueness over D using 2ell rows. */
        long long false_count = 0;
        int vals2ell[ROWS];
        for (int t = 0; t < mD.size; t++) {
            projection_digits(mD.v[t], ROWS, vals2ell);
            int equal = 1;
            for (int j = 0; j < ROWS; j++) {
                if (vals2ell[j] != s_digits[j]) { equal = 0; break; }
            }
            if (equal && !vec_equal(mD.v[t], s)) false_count++;
        }
        false_sum += (double)false_count;
        if (false_count == 0) unique_sum += 1.0;

        /* [2], [4] Layer-1 collisions for T1,T2. */
        memset(hist_T1, 0, (size_t)range * sizeof(int));
        memset(hist_T2_random, 0, (size_t)range * sizeof(int));
        memset(hist_T2_correct, 0, (size_t)range * sizeof(int));
        memset(hist_rand1, 0, (size_t)range * sizeof(int));
        memset(hist_rand2, 0, (size_t)range * sizeof(int));
        for (int b = 0; b < range; b++) {
            head_T1[b] = -1;
            head_T2_correct[b] = -1;
        }

        int vals[ELL];
        for (int t = 0; t < mT[0].size; t++) {
            projection_digits(mT[0].v[t], ELL, vals);
            int idx = encode_digits(vals, ELL);
            hist_T1[idx]++;
            next_T1[t] = head_T1[idx];
            head_T1[idx] = t;
            hist_rand1[rnd_int(range)]++;
        }
        for (int t = 0; t < mT[1].size; t++) {
            projection_digits(mT[1].v[t], ELL, vals);
            int idx_random = encode_difference(r_random, vals, ELL);
            int idx_correct = encode_difference(a1_digits, vals, ELL);
            hist_T2_random[idx_random]++;
            hist_T2_correct[idx_correct]++;
            next_T2_correct[t] = head_T2_correct[idx_correct];
            head_T2_correct[idx_correct] = t;
            hist_rand2[rnd_int(range)]++;
        }

        long long c_lower_lwe = collision_count(hist_T1, hist_T2_random, range);
        long long c_lower_rand = collision_count(hist_rand1, hist_rand2, range);
        long long c_lower_correct = collision_count(hist_T1, hist_T2_correct, range);
        lower_lwe_sum += (double)c_lower_lwe;
        lower_lwe_sq += (double)c_lower_lwe * (double)c_lower_lwe;
        lower_rand_sum += (double)c_lower_rand;
        lower_rand_sq += (double)c_lower_rand * (double)c_lower_rand;
        lower_correct_sum += (double)c_lower_correct;

        /* [3] E_r for the middle layer: r hits pi(A a1) for some representation a1+a2=s. */
        memset(seen, 0, (size_t)range * sizeof(unsigned char));
        int distinct = 0;
        for (int i = 0; i < R1_s; i++) {
            int left_idx = R1_s_left[i];
            projection_digits(mD1.v[left_idx], ELL, vals);
            int idx = encode_digits(vals, ELL);
            if (!seen[idx]) {
                seen[idx] = 1;
                distinct++;
            }
        }
        Er_middle_sum += (double)distinct / (double)range;

        /* [4] Distinguished-collision sampling among correct-r lower-layer collisions. */
        if (c_lower_correct > 0 && R2_a1 > 0) {
            build_collision_bucket_distribution(range);
            for (int rep = 0; rep < HIT_REPEATS_PER_TRIAL; rep++) {
                int samples = 0;
                while (1) {
                    samples++;
                    if (sample_collision_is_distinguished(&mT[0], &mT[1], c_lower_correct, a1)) break;
                }
                hit_sum += (double)samples;
                hit_sq += (double)samples * (double)samples;
                hit_trials++;
            }
        }

        trials++;

        if (trials % PROGRESS_EVERY_TRIALS == 0 || trials == total_trials) {
            printf("Progress: %lld/%lld trials completed\n", trials, total_trials);
            fflush(stdout);
        }
    }

    double mean_false = false_sum / (double)trials;
    double theory_false = ((double)mD.size - 1.0) / (double)range2;
    double theory_unique = exp(-theory_false);

    double theory_lower = ((double)mT[0].size * (double)mT[1].size) / (double)range;
    double lower_lwe_mean = lower_lwe_sum / (double)trials;
    double lower_rand_mean = lower_rand_sum / (double)trials;
    double lower_lwe_std = sqrt(fmax(0.0, lower_lwe_sq / (double)trials - lower_lwe_mean * lower_lwe_mean));
    double lower_rand_std = sqrt(fmax(0.0, lower_rand_sq / (double)trials - lower_rand_mean * lower_rand_mean));

    double Er_emp = Er_middle_sum / (double)trials;
    double Er_exact = 1.0 - pow(1.0 - 1.0 / (double)range, (double)R1_s);
    double Er_linear = (double)R1_s / (double)range;

    double correct_mean = lower_correct_sum / (double)trials;
    double theory_correct = (double)R2_a1 + (((double)mT[0].size * (double)mT[1].size - (double)R2_a1) / (double)range);
    double hit_mean = hit_sum / (double)hit_trials;
    double hit_std = sqrt(fmax(0.0, hit_sq / (double)hit_trials - hit_mean * hit_mean));
    double theory_hit = theory_correct / (double)R2_a1;

    printf("\nFinished. Trials: %lld.\n", trials);

    printf("\n[1] Projected uniqueness over D using 2ell projected coordinates\n");
    printf("    Theory: E[false projected solutions] = (|D|-1)/q^(2ell).\n");
    printf("    empirical avg false hits    %.8f\n", mean_false);
    printf("    theory avg false hits       %.8f\n", theory_false);
    printf("    empirical unique fraction   %.8f\n", unique_sum / (double)trials);
    printf("    Poisson unique heuristic    %.8f\n", theory_unique);

    printf("\n[2] Layer-1 cross-collision count for T1,T2 with random r\n");
    printf("    Theory: E[# collisions] = |T1||T2|/q^ell.\n");
    printf("    source              empirical_mean      std_dev        theory       relative_error\n");
    printf("    -------------------------------------------------------------------------------\n");
    printf("    LWE projections     %15.6f  %11.6f  %12.6f  %15.6f\n",
           lower_lwe_mean, lower_lwe_std, theory_lower, (lower_lwe_mean - theory_lower) / theory_lower);
    printf("    random functions    %15.6f  %11.6f  %12.6f  %15.6f\n",
           lower_rand_mean, lower_rand_std, theory_lower, (lower_rand_mean - theory_lower) / theory_lower);

    printf("\n[3] Middle-layer event E_r\n");
    printf("    E_r means random r hits pi(A a1) for at least one of the R1 representations a1+a2=s.\n");
    printf("    R1(s)                          %d\n", R1_s);
    printf("    empirical Pr[E_r]              %.8f\n", Er_emp);
    printf("    theory exact independent       %.8f\n", Er_exact);
    printf("    theory linear R1/q^ell         %.8f\n", Er_linear);

    printf("\n[4] Distinguished-collision ratio for lower layer a1=s1+s2\n");
    printf("    Here r is set to pi(A a1), so the R2(a1) planted representations are guaranteed collisions.\n");
    printf("    Theory total = R2 + (|T1||T2|-R2)/q^ell; predicted samples = total/R2.\n");
    printf("    R2(a1)                         %d\n", R2_a1);
    printf("    empirical avg total collisions %.6f\n", correct_mean);
    printf("    theory total collisions        %.6f\n", theory_correct);
    printf("    empirical mean samples         %.6f\n", hit_mean);
    printf("    empirical std. dev.            %.6f\n", hit_std);
    printf("    theory mean samples            %.6f\n", theory_hit);

    printf("\nDone.\n");

    free(R2_a1_left);
    free(R2_a2_left);
    free(R1_s_left);
    free(hist_T1); free(hist_T2_random); free(hist_T2_correct); free(hist_rand1); free(hist_rand2);
    free(head_T1); free(head_T2_correct); free(next_T1); free(next_T2_correct);
    free(seen); free(bucket_id); free(bucket_cum);
    free(B.v); free(Jlower.v); free(Jmiddle.v); free(Jsecret.v);
    for (int i = 0; i < NUM_BLOCKS; i++) { free((void *)mT[i].name); free(mT[i].v); }
    free(mD.v); free(mD1.v); free(mD2.v);

    return 0;
}
