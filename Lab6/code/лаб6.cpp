#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
using namespace std;

const double EPS = 0.001, ALPHA = 0.5, H = 1e-5;

double f(double x) { return (2 - x * x) * sin(x + 0.5) - (2 * cos(x + 0.5) + 1) * x; }
double df(double x) { return -x * x * cos(x + 0.5) - 1; }
double d2f(double x) { return -2 * x * cos(x + 0.5) + x * x * sin(x + 0.5); }

// Класичний Ньютон
bool newton(double& x, int& n, bool print)
{
    if (print)
        cout << "NEWTON\n" << left << setw(5) << "k" << setw(13) << "xk"
        << setw(13) << "f(xk)" << setw(13) << "f'(xk)" << setw(13) << "f''(xk)" << "\n"
        << string(57, '-') << "\n";

    for (n = 0; n < 50; n++)
    {
        double d1 = df(x), d2 = d2f(x);

        if (print)
            cout << left << setw(5) << n << setw(13) << x << setw(13) << f(x)
            << setw(13) << d1 << setw(13) << d2 << "\n";

        if (fabs(d1) <= EPS) return true;
        if (d2 <= 1e-12) return false;

        x -= d1 / d2;
    }
    return true;
}

// Ньютон-Рафсон 
bool newtonRaphson(double& x, int& n, bool print)
{
    if (print)
        cout << "\nNEWTON-RAPHSON\n" << left << setw(5) << "k" << setw(13) << "xk"
        << setw(13) << "f(xk)" << setw(13) << "f'(xk)" << setw(13) << "f''(xk)"
        << setw(10) << "t" << setw(13) << "xbar" << setw(13) << "f(xbar)"
        << setw(13) << "f'(xbar)" << "Action\n" << string(120, '-') << "\n";

    double fx = f(x), d1 = df(x), d2 = d2f(x);
    if (d2 <= 1e-12) return false;

    for (n = 0; n < 50 && fabs(d1) > EPS; )
    {
        double t = 1.0;
        bool ok = false;

        for (int a = 0; a < 30 && !ok; a++, t /= 2)
        {
            double xb = x - t * d1 / d2;
            double d1b = df(xb), fb = f(xb);
            bool stop = fabs(d1b) <= EPS;
            ok = stop || (fb - fx <= -ALPHA * t * d1 * d1 / d2);

            if (print)
                cout << left << setw(5) << n + 1 << setw(13) << x << setw(13) << fx
                << setw(13) << d1 << setw(13) << d2 << setw(10) << t
                << setw(13) << xb << setw(13) << fb << setw(13) << d1b
                << (stop ? "stop" : ok ? "ok" : "fail") << "\n";

            if (ok)
            {
                x = xb; fx = fb; d1 = d1b;
                if (!stop) d2 = d2f(x);
                n++;
            }
        }
        if (!ok) return false;
    }
    return true;
}

void summary(double pts[], int cnt)
{
    cout << "\nSUMMARY\n" << left << setw(10) << "x0" << setw(10) << "Method"
        << setw(13) << "x*" << setw(13) << "f(x*)" << setw(13) << "|f'(x*)|" << "n\n"
        << string(65, '-') << "\n";

    for (int i = 0; i < cnt; i++)
        for (int m = 0; m < 2; m++)
        {
            double x = pts[i];
            int n;
            bool ok = m ? newtonRaphson(x, n, false) : newton(x, n, false);

            cout << left << setw(10) << pts[i] << setw(10) << (m ? "N-R" : "Newton");
            if (ok)
                cout << setw(13) << x << setw(13) << f(x) << setw(13) << fabs(df(x)) << n << "\n";
            else
                cout << "-\n";
        }
}

int main()
{
    cout << fixed << setprecision(4);

    double a0 = 1.45;
    double num = (f(a0 + H) - f(a0 - H)) / (2 * H);
    cout << "f'(x) analytical = " << df(a0) << "\nf'(x) numerical  = " << num
        << "\nDifference       = " << fabs(df(a0) - num) << "\n\n";

    double x = a0; int n;
    newton(x, n, true);

    x = a0;
    newtonRaphson(x, n, true);

    double pts[] = { 1.45, 1.75, 5.0 };
    summary(pts, 3);
}