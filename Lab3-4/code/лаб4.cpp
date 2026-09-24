#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace std;

double f(double x)
{
    return (2 - x * x) * sin(x + 0.5) - (2 * cos(x + 0.5) + 1) * x;
}

int main()
{
    cout << fixed << setprecision(4);

    double a = 1.45;
    double b = 1.75;
    double eps = 0.01;
    double a0 = a;
    double b0 = b;
    double L0 = b0 - a0;
    double R = L0 / eps;

    cout << "Method of Fibonacci" << endl;

    cout << "f(x) = (2 - x^2)sin(x + 0.5) - "
        << "(2cos(x + 0.5) + 1)x" << endl;

    cout << "\na0 = " << a0 << endl;
    cout << "b0 = " << b0 << endl;
    cout << "epsilon = " << eps << endl;
    cout << "L0 = " << L0 << endl;
    cout << "R = L0 / epsilon = " << R << endl;

    vector<long long> F;

    F.push_back(0); 
    F.push_back(1); 
    F.push_back(1); 

    int N = 2;

    cout << "\nTable of Fibonacci numbers:" << endl;
    cout << "N\tF[N]\tF[N] >= R" << endl;

    while (F[N] < R)
    {
        F.push_back(F[N] + F[N - 1]);
        N++;

        cout << N << "\t"
            << F[N] << "\t";

        if (F[N] >= R)
            cout << "YES";
        else
            cout << "NO";

        cout << endl;
    }

    cout << "\nN = " << N << endl;
    cout << "F[N] = " << F[N] << endl;

    cout << "\nFibonacci method table:" << endl;

    cout << setw(4) << "k"
        << setw(6) << "m"
        << setw(12) << "a_k"
        << setw(12) << "b_k"
        << setw(12) << "x1"
        << setw(12) << "x2"
        << setw(12) << "f(x1)"
        << setw(12) << "f(x2)"
        << setw(12) << "Lk"
        << setw(14) << "decision"
        << endl;

    int m = N;
    double x1 = a + (double)F[m - 2] / F[m] * (b - a);
    double x2 = a + (double)F[m - 1] / F[m] * (b - a);
    double f1 = f(x1);
    double f2 = f(x2);
    int functionCalculations = 2;

    for (int k = 0; k < N - 2; k++)
    {
        m = N - k;

        cout << setw(4) << k
            << setw(6) << m
            << setw(12) << a
            << setw(12) << b
            << setw(12) << x1
            << setw(12) << x2
            << setw(12) << f1
            << setw(12) << f2
            << setw(12) << b - a;

        if (f1 <= f2)
        {
            cout << setw(14) << "b = x2" << endl;

            b = x2;
            x2 = x1;
            f2 = f1;

            if (m > 3)
            {
                x1 = a + (double)F[m - 3] / F[m - 1] * (b - a);
                f1 = f(x1);
                functionCalculations++;
            }
        }
        else
        {
            cout << setw(14) << "a = x1" << endl;

            a = x1;
            x1 = x2;
            f1 = f2;

            if (m > 3)
            {
                x2 = a + (double)F[m - 2] / F[m - 1] * (b - a);
                f2 = f(x2);

                functionCalculations++;
            }
        }
    }

    double xStar = (a + b) / 2.0;
    double Lfinal = b - a;

    cout << "\nResult:" << endl;

    cout << "x* = " << xStar << endl;
    cout << "f(x*) = " << f(xStar) << endl;
    cout << "L_final = " << Lfinal << endl;

    cout << "\nNumber of function calculations = "
        << functionCalculations << endl;

    double guarantee = L0 / F[N];

    cout << "\nGuarantee check:" << endl;

    cout << "L0 / F[N] = " << guarantee << endl;
    cout << "L_final = " << Lfinal << endl;

    if (Lfinal <= guarantee)
    {
        cout << "L_final <= L0 / F[N] : YES" << endl;
    }
    else
    {
        cout << "L_final <= L0 / F[N] : NO" << endl;
    }
}