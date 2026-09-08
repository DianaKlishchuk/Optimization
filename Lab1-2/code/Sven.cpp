#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace std;

double f(double x) {
    return (2.0 - x * x) * sin(x + 0.5) - (2.0 * cos(x + 0.5) + 1.0) * x;
}

int main() {
    setlocale(LC_ALL, "");
    double x0 = 1.45;
    double delta = 0.1;
    double f0 = f(x0);
    double x1 = x0 + delta;
    double f1 = f(x1);

    cout << fixed << setprecision(6);

    cout << "Method Sven\n";
    cout << "x0 = " << x0 << ",  delta = " << delta << "\n\n";

    cout << left
        << setw(6) << "k"
        << setw(12) << "delta"
        << setw(12) << "x_k"
        << setw(14) << "f(x_k)"
        << setw(10) << "f(x_k) < f(previous)"
        << "\n";
    cout << string(54, '-') << "\n";

    cout << setw(6) << 0
        << setw(12) << delta
        << setw(12) << x0
        << setw(14) << f0
        << setw(10) << "-"
        << "\n";
    cout << setw(6) << 1
        << setw(12) << delta
        << setw(12) << x1
        << setw(14) << f1
        << setw(10) << (f1 < f0 ? "+" : "-")
        << "\n";

    vector<double> xs = { x0, x1 };
    vector<double> fs = { f0, f1 };
    int k = 1;
    double xk = x1;
    double fk = f1;
    double step = pow(2.0, k) * delta;
    double xk1 = xk + step;
    double fk1 = f(xk1);

    while (fk1 < fk) {
        xs.push_back(xk1);
        fs.push_back(fk1);
        xk = xk1;
        fk = fk1;
        k++;
        step = pow(2.0, k) * delta;
        xk1 = xk + step;
        fk1 = f(xk1);
        cout << setw(6) << k + 1
            << setw(12) << delta
            << setw(12) << xk1
            << setw(14) << fk1
            << setw(10) << "+"
            << "\n";
    }
    xs.push_back(xk1);
    fs.push_back(fk1);

    cout << setw(6) << k + 1
        << setw(12) << delta
        << setw(12) << xk1
        << setw(14) << fk1
        << setw(10) << "-"
        << "\n";

    int n = (int)xs.size() - 1;
    double a0 = xs[n - 2];
    double b0 = xs[n];

    if (a0 > b0)
        swap(a0, b0);

    cout << "\nInterval:\n";
    cout << "[a0, b0] = [" << a0 << "; " << b0 << "]\n";
    cout << "L0 = " << (b0 - a0) << "\n";
}