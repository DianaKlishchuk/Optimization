#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

double f(double x) {
    return (2.0 - x * x) * sin(x + 0.5) - (2.0 * cos(x + 0.5) + 1.0) * x;
}

int main() {
    setlocale(LC_ALL, "");
    double a = 1.45;
    double b = 1.75;
    double sigma = 0.1;
    double eps = 0.01;
    int k = 0;

    cout << fixed << setprecision(6);
    cout << "Method dichotomy\n";
    cout << "Interval: [" << a << "; " << b << "]\n";
    cout << "sigma = " << sigma << ", epsilon = " << eps << "\n\n";

    cout << left
        << setw(5) << "k"
        << setw(12) << "x1"
        << setw(12) << "x2"
        << setw(14) << "f(x1)"
        << setw(14) << "f(x2)"
        << setw(18) << "[ak, bk]"
        << setw(10) << "Lk"
        << "\n";
    cout << string(85, '-') << "\n";

    cout << setw(5) << k
        << setw(12) << "-"
        << setw(12) << "-"
        << setw(14) << "-"
        << setw(14) << "-"
        << "[" << a << "; " << b << "]"
        << setw(10) << b - a
        << "\n";

    while ((b - a) > sigma) {

        double x1 = (a + b) / 2.0 - eps / 2.0;
        double x2 = (a + b) / 2.0 + eps / 2.0;
        double f1 = f(x1);
        double f2 = f(x2);

        if (f1 > f2) {
            a = x1;
        }
        else {
            b = x2;
        }

        k++;

        cout << setw(5) << k
            << setw(12) << x1
            << setw(12) << x2
            << setw(14) << f1
            << setw(14) << f2
            << "[" << a << "; " << b << "]"
            << setw(10) << b - a
            << "\n";
    }

    double xmin = (a + b) / 2.0;
    double fmin = f(xmin);

    cout << "x* = " << xmin << "\n";
    cout << "f(x*) = " << fmin << "\n";
    cout << "Final interval: [" << a << "; " << b << "]\n";
    cout << "L = " << b - a << " <= sigma = " << sigma << "\n";
}