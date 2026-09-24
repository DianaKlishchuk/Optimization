#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

double f(double x) {
    return (2.0 - x * x) * sin(x + 0.5) - (2.0 * cos(x + 0.5) + 1.0) * x;
}

int main() {

    double a = 1.45;
    double b = 1.75;
    double eps = 0.01;
    double tau = (sqrt(5.0) - 1.0) / 2.0;
    double x1 = a + (1.0 - tau) * (b - a);
    double x2 = a + tau * (b - a);
    double f1 = f(x1);
    double f2 = f(x2);
    int Nf = 2;
    int k = 0;

    cout << fixed << setprecision(4);
    cout << "GOLDEN SECTION METHOD\n\n";

    cout << setw(4) << "k"
        << setw(12) << "a"
        << setw(12) << "b"
        << setw(12) << "x1"
        << setw(12) << "x2"
        << setw(14) << "f(x1)"
        << setw(14) << "f(x2)"
        << setw(12) << "Lk"
        << setw(12) << "Lk+1/Lk"
        << "\n";

    cout << string(116, '-') << "\n";

    while (b - a > eps) {

        double L = b - a;

        cout << setw(4) << k
            << setw(12) << a
            << setw(12) << b
            << setw(12) << x1
            << setw(12) << x2
            << setw(14) << f1
            << setw(14) << f2
            << setw(12) << L;

        if (f1 <= f2) {

            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + (1.0 - tau) * (b - a);
            f1 = f(x1);
            Nf++;

        }
        else {

            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + tau * (b - a);
            f2 = f(x2);
            Nf++;
        }

        k++;

        cout << setw(12) << (b - a) / L
            << "\n";
    }

    double x_star = (a + b) / 2.0;
    double f_star = f(x_star);

    cout << "\nFinal interval: ["
        << a << "; " << b << "]\n";

    cout << "Final interval length L = "
        << b - a << "\n";

    cout << "Number of iterations = "
        << k << "\n";

    cout << "Number of function evaluations Nf = "
        << Nf << "\n";

    cout << "Minimum point x* = "
        << x_star << "\n";

    cout << "Minimum value f(x*) = "
        << f_star << "\n";
}