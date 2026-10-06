#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

double f(double x)
{
    return (2 - x * x) * sin(x + 0.5)
        - (2 * cos(x + 0.5) + 1) * x;
}

int main()
{
    double x1 = 1.45;
    double dx = 0.1;
    double s_f = 0.01;
    double s_x = 0.001;
    double x2 = x1 + dx;
    double f1 = f(x1);
    double f2 = f(x2);
    double x3;

    if (f1 > f2)
        x3 = x1 + 2 * dx;
    else
        x3 = x1 - dx;

    double f3 = f(x3);
    int n = 0;

    cout << fixed << setprecision(6);
    cout << setw(3) << "k"
        << setw(11) << "x1"
        << setw(11) << "x2"
        << setw(11) << "x3"
        << setw(13) << "f(x1)"
        << setw(13) << "f(x2)"
        << setw(13) << "f(x3)"
        << setw(11) << "a1"
        << setw(11) << "a2"
        << setw(11) << "xt"
        << setw(13) << "f(xt)"
        << setw(16) << "|fmin-f(xt)|"
        << setw(14) << "|xmin-xt|"
        << setw(25) << "Action"
        << '\n';

    cout << string(190, '-') << '\n';

    for (int k = 1; k <= 50; k++)
    {
        n++;
        double xs[3] = { x1, x2, x3 };
        double fs[3] = { f1, f2, f3 };
        int iMin = 0;
        int iMax = 0;

        for (int i = 1; i < 3; i++)
        {
            if (fs[i] < fs[iMin])
                iMin = i;

            if (fs[i] > fs[iMax])
                iMax = i;
        }

        if (fabs(x1 - x2) <= 1e-12 ||
            fabs(x1 - x3) <= 1e-12 ||
            fabs(x2 - x3) <= 1e-12)
        {
            cout << "\nError: two points coincide.\n";
            cout << "Change dx and restart.\n";
        }

        double a1 = (f2 - f1) / (x2 - x1);
        double a2 = ((f3 - f1) / (x3 - x1) - a1) / (x3 - x2);
        double xt;

        if (a2 > 0)
        {
            xt = -a1 / (2 * a2) + (x1 + x2) / 2;
        }
        else if (a2 < 0)
        {
            xt = 2 * xs[iMin] - xs[iMax];
        }
        else
        {
            cout << "\nError: a2 = 0.\n";
            cout << "Change dx and restart.\n";
        }

        double ft = f(xt);
        double df = fabs(fs[iMin] - ft);
        double dxError = fabs(xs[iMin] - xt);
        string action;

        if (df <= s_f && dxError <= s_x)
        {
            action = "STOP";
        }
        else if (fabs(xt - x1) <= 1e-12 ||
            fabs(xt - x2) <= 1e-12 ||
            fabs(xt - x3) <= 1e-12)
        {
            action = "xt coincides";
        }
        else if (a2 < 0)
        {
            action = "replace xmax";
        }
        else
        {
            action = "replace xmax";
        }

        cout << setw(3) << k
            << setw(11) << x1
            << setw(11) << x2
            << setw(11) << x3
            << setw(13) << f1
            << setw(13) << f2
            << setw(13) << f3
            << setw(11) << a1
            << setw(11) << a2
            << setw(11) << xt
            << setw(13) << ft
            << setw(16) << df
            << setw(14) << dxError
            << setw(25) << action
            << '\n';

        if (df <= s_f && dxError <= s_x)
        {
            cout << "\nResult:\n";
            cout << "x* = " << xt << '\n';
            cout << "f* = " << ft << '\n';
            cout << "Iterations n = " << n << '\n';
            cout << "Function evaluations Nf = " << n + 3 << '\n';
            break;
        }

        xs[iMax] = xt;
        fs[iMax] = ft;
        x1 = xs[0];
        x2 = xs[1];
        x3 = xs[2];
        f1 = fs[0];
        f2 = fs[1];
        f3 = fs[2];
    }

}