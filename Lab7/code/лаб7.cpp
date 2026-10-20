#include <cmath>
#include <cstdio>
using namespace std;

double f(double x) { return (2 - x * x) * sin(x + 0.5) - (2 * cos(x + 0.5) + 1) * x; }
double df(double x) { return -x * x * cos(x + 0.5) - 1; }

struct Result {
    double x_star;
    int iters;    
    int n_df;    
    bool ok;
};

Result secant(double a0, double b0, double eps, int mode, bool print_table) {
    Result r{ 0, 0, 0, false };

    double xm = a0, xk = b0;          
    double pm = df(xm), pk = df(xk);       
    r.n_df = 2;

    if (mode == 0 && pm * pk >= 0) {
        printf("f'(x0)*f'(x1) >= 0: derivative does not change sign on [a0, b0]\n");
        return r;
    }

    if (print_table) {
        printf("%3s %11s %12s %11s %12s %11s %12s %12s  %s\n",
            "k", "x_(k-1)", "f'(x_(k-1))", "x_k", "f'(x_k)",
            "x_(k+1)", "f'(x_(k+1))", "f(x_(k+1))", "action");
    }

    double xn = xk;
    for (int k = 1; k <= 100; ++k) {
        if (fabs(pk - pm) <= 1e-12) {     
            printf("f'(x_k) = f'(x_(k-1)): secant is horizontal, choose another interval\n");
            return r;
        }
        xn = (pk * xm - pm * xk) / (pk - pm);   
        double pn = df(xn);                  
        ++r.n_df;
        r.iters = k;
        const char* action = "";
        bool stop = fabs(pn) <= eps;

        if (!stop) action = (mode == 0) ? (pn > 0 ? "x_k <- x_(k+1)" : "x_(k-1) <- x_(k+1)")
            : "shift pair";

        if (print_table) {
            printf("%3d %11.6f %12.6f %11.6f %12.6f %11.6f %12.6f %12.6f  %s\n",
                k, xm, pm, xk, pk, xn, pn, f(xn), stop ? "|f'| <= eps, STOP" : action);
        }
        if (stop) { r.ok = true; break; }

        if (mode == 0) {                  
            if (pn > 0) { xk = xn; pk = pn; } 
            else { xm = xn; pm = pn; }
        }
        else {                                
            xm = xk; pm = pk;
            xk = xn; pk = pn;
        }
    }
    r.x_star = xn;
    return r;
}

int main() {
    double a0 = 1.45, b0 = 1.75, eps = 0.001;   

    printf("f'(a0) = %.6f, f'(b0) = %.6f, product = %.6f\n\n", df(a0), df(b0), df(a0) * df(b0));

    printf("Textbook scheme \n");
    Result r1 = secant(a0, b0, eps, 0, true);
    if (r1.ok)
        printf("\nx* = %.6f, f(x*) = %.6f, n = %d, N_f' = %d (n + 2), f'': 0\n",
            r1.x_star, f(r1.x_star), r1.iters, r1.n_df);

    printf("\nPure secant (experiment) \n");
    Result r2 = secant(a0, b0, eps, 1, true);
    if (r2.ok)
        printf("\nx* = %.6f, f(x*) = %.6f, n = %d, N_f' = %d\n",
            r2.x_star, f(r2.x_star), r2.iters, r2.n_df);

    if (r1.ok && r2.ok)
        printf("\nDifference in x* between the two variants: %.3e\n", fabs(r1.x_star - r2.x_star));
    return 0;
}