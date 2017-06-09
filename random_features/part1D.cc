#include <map>
#include <iterator>
#include <mex.h>
#include <matrix.h>

using namespace std;

typedef map<int,int> map_type;

template <class MAP>
int addids(MAP &pdb, int *i, int *end, int *id)
{
	int classid=0;

	for(; i!=end; ++i, ++id) {
		typename MAP::iterator item = pdb.find(*i);
		if(item==pdb.end())
		{
			pdb.insert(pair<int,int>(*i,classid));
			*id = classid;
			++classid;
		} else
			*id = item->second;
	}

	return classid;
}

void
mexFunction(int nlhs, mxArray *plhs[],
			int nrhs, const mxArray *prhs[])
{
	if(nrhs!=1)
		mexErrMsgTxt("part1D(vector)");

	const mxArray *X = prhs[0];
	if(mxGetClassID(X)!=mxINT32_CLASS)
		mexErrMsgTxt("vector must be int32.");

	int N = mxGetN(X)*mxGetM(X);

	map_type pdb;

	mxArray *out = mxCreateNumericMatrix(1,N,mxINT32_CLASS,mxREAL);
	addids(pdb,
		   static_cast<int*>(mxGetData(X)),
		   static_cast<int*>(mxGetData(X))+N,
		   static_cast<int*>(mxGetData(out)));
	plhs[0] = out;

	if(nlhs>1) {
		const int nentries = pdb.size();
		mxArray *tab = mxCreateNumericMatrix(2,nentries,mxINT32_CLASS,mxREAL);
		int *tab_data = static_cast<int*>(mxGetData(tab));

		map_type::iterator it = pdb.begin();

		for(int i=0; i<nentries; ++i, ++it) {
			tab_data[2*i+0] = it->first;
			tab_data[2*i+1] = it->second;
			if(it == pdb.end()) mexErrMsgTxt("internal error");
		}
		plhs[1] = tab;
	}
}


/*
#include <iostream>
int main()
{
	const int X[] = {10, 20, 20, 30, 10, 30, 20, 30, 10};
	const int n = sizeof(X)/sizeof(X[0]);
	int id[n];

	squeezeid(X,X+n,id);

	for(int i=0;i<n; ++i)
		cout << id[i] << ' ';
	cout << '\n';
}
*/
