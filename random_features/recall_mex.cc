#include <map>
#include <cassert>
#include <algorithm>
#include <matrix.h>
#include <mex.h>

using namespace std;


int binary_search(int N, const int *tab, int x)
{
	const int *ptr = lower_bound(tab,tab+N,x);
	if(ptr==tab+N)
		return -1;
	if(*ptr!=x)
		return -1;
	return ptr-tab;
}

void
lookup(int Ntab, const int *tab, int N, const int *x, int *out)
{
	for(int i=0; i<N; ++i) {
		int v = binary_search(Ntab,tab,x[i]);
		out[i] = (v>=0)?tab[Ntab+v]:-1;
	}
}


void
mexFunction(int nlhs, mxArray *plhs[],
			int nrhs, const mxArray *prhs[])
{
	if(nrhs!=2)
		mexErrMsgTxt("recall(table,x)");

	const mxArray *tab = prhs[0];
	const mxArray *x = prhs[1];
	if(mxGetClassID(tab)!=mxINT32_CLASS)
		mexErrMsgTxt("table must be int32.");
	if(mxGetClassID(x)!=mxINT32_CLASS)
		mexErrMsgTxt("x must be int32.");
	if(mxGetN(tab)!=2)
		mexErrMsgTxt("table must be Nx2.");

	int Ntab = mxGetM(tab);
	int N = mxGetM(x)*mxGetN(x);

	mxArray *out = mxCreateNumericMatrix(1,N,mxINT32_CLASS,mxREAL);
	lookup(Ntab, static_cast<const int*>(mxGetData(tab)),
		   N, static_cast<const int*>(mxGetData(x)),
		   static_cast<int*>(mxGetData(out)));
	plhs[0] = out;
}
