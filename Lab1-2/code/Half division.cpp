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
    double sigma = 0.1;
    int k = 0;

    cout << fixed << setprecision(6);
    cout << "Method Half division\n\n";
    cout << setw(5) << "k"
        << setw(12) << "x1"
        << setw(12) << "xm"
        << setw(12) << "x2"
        << setw(14) << "f(x1)"
        << setw(14) << "f(xm)"
        << setw(14) << "f(x2)"
        << setw(12) << "Lk"
        << "\n";
    cout << string(95, '-') << "\n";
    cout << setw(5) << 0
        << setw(12) << "-"
        << setw(12) << "-"
        << setw(12) << "-"
        << setw(14) << "-"
        << setw(14) << "-"
        << setw(14) << "-"
        << setw(12) << b - a
        << "\n";

    while (b - a > sigma) {

        double L = b - a;
        double xm = (a + b) / 2.0;
        double x1 = a + L / 4.0;
        double x2 = b - L / 4.0;
        double f1 = f(x1);
        double fm = f(xm);
        double f2 = f(x2);

        cout << setw(5) << k + 1
            << setw(12) << x1
            << setw(12) << xm
            << setw(12) << x2
            << setw(14) << f1
            << setw(14) << fm
            << setw(14) << f2
            << setw(12) << L
            << "\n";

        if (fm <= f1 && fm <= f2) {
            a = x1;
            b = x2;
        }
        else if (f1 < fm) {
            b = xm;
        }
        else {
            a = xm;
        }

        k++;
    }

    double x_min = (a + b) / 2.0;
    double f_min = f(x_min);

    cout << "\nFinal interval: ["
        << a << "; " << b << "]\n";

    cout << "Final interval length L = "
        << b - a << "\n";

    cout << "Minimum point x* = "
        << x_min << "\n";

    cout << "Minimum value f(x*) = "
        << f_min << "\n";

}