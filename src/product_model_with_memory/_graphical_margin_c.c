#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>

typedef struct { int y, index; } ActivePair;

static int compare_active_pair(const void *left, const void *right) {
    const ActivePair *a=(const ActivePair*)left, *b=(const ActivePair*)right;
    return (a->y>b->y)-(a->y<b->y);
}

static PyObject *intersection_plan(PyObject *self, PyObject *args) {
    PyArrayObject *edge_a, *edge_b, *ya_y, *ya_a, *yb_y, *yb_b;
    long long maximum=-1;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!|L",
            &PyArray_Type,&edge_a, &PyArray_Type,&edge_b,
            &PyArray_Type,&ya_y, &PyArray_Type,&ya_a,
            &PyArray_Type,&yb_y, &PyArray_Type,&yb_b, &maximum)) return NULL;
    npy_intp ne=PyArray_SIZE(edge_a), n1=PyArray_SIZE(ya_y),
             n2=PyArray_SIZE(yb_y), v=0, total=0;
    int *ea=PyArray_DATA(edge_a), *eb=PyArray_DATA(edge_b);
    int *ay=PyArray_DATA(ya_y), *aa=PyArray_DATA(ya_a);
    int *by=PyArray_DATA(yb_y), *bb=PyArray_DATA(yb_b);
    for(npy_intp i=0;i<ne;i++){if(ea[i]+1>v)v=ea[i]+1;if(eb[i]+1>v)v=eb[i]+1;}
    for(npy_intp i=0;i<n1;i++)if(aa[i]+1>v)v=aa[i]+1;
    for(npy_intp i=0;i<n2;i++)if(bb[i]+1>v)v=bb[i]+1;
    npy_intp *p1=calloc((size_t)v+1,sizeof(*p1));
    npy_intp *p2=calloc((size_t)v+1,sizeof(*p2));
    npy_intp *cursor=malloc(((size_t)v+1)*sizeof(*cursor));
    ActivePair *r1=malloc((size_t)n1*sizeof(*r1));
    ActivePair *r2=malloc((size_t)n2*sizeof(*r2));
    PyArrayObject *oe=NULL,*oy=NULL,*o1=NULL,*o2=NULL;
    if(!p1||!p2||!cursor||(!r1&&n1)||(!r2&&n2)){PyErr_NoMemory();goto fail_plan;}
    for(npy_intp i=0;i<n1;i++)p1[aa[i]+1]++;
    for(npy_intp i=0;i<n2;i++)p2[bb[i]+1]++;
    for(npy_intp a=0;a<v;a++){p1[a+1]+=p1[a];p2[a+1]+=p2[a];}
    memcpy(cursor,p1,((size_t)v+1)*sizeof(*cursor));
    for(npy_intp i=0;i<n1;i++){npy_intp j=cursor[aa[i]]++;r1[j]=(ActivePair){ay[i],(int)i};}
    memcpy(cursor,p2,((size_t)v+1)*sizeof(*cursor));
    for(npy_intp i=0;i<n2;i++){npy_intp j=cursor[bb[i]]++;r2[j]=(ActivePair){by[i],(int)i};}
    for(npy_intp a=0;a<v;a++){
        qsort(r1+p1[a],(size_t)(p1[a+1]-p1[a]),sizeof(*r1),compare_active_pair);
        qsort(r2+p2[a],(size_t)(p2[a+1]-p2[a]),sizeof(*r2),compare_active_pair);
    }
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp e=0;e<ne;e++){
        npy_intp i=p1[ea[e]],ie=p1[ea[e]+1],j=p2[eb[e]],je=p2[eb[e]+1];
        while(i<ie&&j<je){if(r1[i].y<r2[j].y)i++;else if(r2[j].y<r1[i].y)j++;else{total++;i++;j++;}}
    }
    Py_END_ALLOW_THREADS
    if(maximum>=0 && total>(npy_intp)maximum){PyErr_SetString(PyExc_MemoryError,"intersection plan exceeds limit");goto fail_plan;}
    npy_intp dims[1]={total};
    oe=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_INT32,0);
    oy=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_INT32,0);
    o1=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_INT32,0);
    o2=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_INT32,0);
    if(!oe||!oy||!o1||!o2){PyErr_NoMemory();goto fail_plan;}
    int *de=PyArray_DATA(oe),*dy=PyArray_DATA(oy),*d1=PyArray_DATA(o1),*d2=PyArray_DATA(o2);
    npy_intp out=0;
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp e=0;e<ne;e++){
        npy_intp i=p1[ea[e]],ie=p1[ea[e]+1],j=p2[eb[e]],je=p2[eb[e]+1];
        while(i<ie&&j<je){
            if(r1[i].y<r2[j].y)i++;
            else if(r2[j].y<r1[i].y)j++;
            else{de[out]=(int)e;dy[out]=r1[i].y;d1[out]=r1[i].index;d2[out]=r2[j].index;out++;i++;j++;}
        }
    }
    Py_END_ALLOW_THREADS
    free(p1);free(p2);free(cursor);free(r1);free(r2);
    return Py_BuildValue("NNNN",oe,oy,o1,o2);
fail_plan:
    free(p1);free(p2);free(cursor);free(r1);free(r2);
    Py_XDECREF(oe);Py_XDECREF(oy);Py_XDECREF(o1);Py_XDECREF(o2);
    return NULL;
}

typedef struct {
    npy_intp lo, hi, v, n1, n2;
    const double *b, *q1, *q2, *me;
    const int *ie, *iy, *ii1, *ii2;
    double *cross, *oy, *o1, *o2;
    int phase;
} MarginTask;

static void *margin_worker(void *raw) {
    MarginTask *t=(MarginTask*)raw;
    if(t->phase==0) {
        for(npy_intp i=t->lo;i<t->hi;i++)
            t->cross[t->ie[i]] += t->b[t->iy[i]]*t->q1[t->ii1[i]]*t->q2[t->ii2[i]];
    } else {
        for(npy_intp i=t->lo;i<t->hi;i++) {
            double common=t->b[t->iy[i]]*t->me[t->ie[i]];
            t->o1[t->ii1[i]]+=common*(1+t->q1[t->ii1[i]])*t->q2[t->ii2[i]];
            t->o2[t->ii2[i]]+=common*(1+t->q2[t->ii2[i]])*t->q1[t->ii1[i]];
            t->oy[t->iy[i]]+=common*t->q1[t->ii1[i]]*t->q2[t->ii2[i]];
        }
    }
    return NULL;
}

static PyObject *fused_margins(PyObject *self, PyObject *args) {
    PyArrayObject *base, *r1, *r2, *edge_a, *edge_b, *edge_p;
    PyArrayObject *ya_y, *ya_a, *yb_y, *yb_b;
    PyArrayObject *ix_e, *ix_y, *ix_1, *ix_2;
    int workers=1;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!O!O!|i",
            &PyArray_Type,&base, &PyArray_Type,&r1, &PyArray_Type,&r2,
            &PyArray_Type,&edge_a, &PyArray_Type,&edge_b,
            &PyArray_Type,&edge_p, &PyArray_Type,&ya_y,
            &PyArray_Type,&ya_a, &PyArray_Type,&yb_y,
            &PyArray_Type,&yb_b, &PyArray_Type,&ix_e,
            &PyArray_Type,&ix_y, &PyArray_Type,&ix_1,
            &PyArray_Type,&ix_2, &workers)) return NULL;
    if(workers<1) workers=1;

    npy_intp v=PyArray_SIZE(base), n1=PyArray_SIZE(r1), n2=PyArray_SIZE(r2);
    npy_intp ne=PyArray_SIZE(edge_p), ni=PyArray_SIZE(ix_e), dims[1];
    dims[0]=v; PyArrayObject *my=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_DOUBLE,0);
    dims[0]=n1; PyArrayObject *m1=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_DOUBLE,0);
    dims[0]=n2; PyArrayObject *m2=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_DOUBLE,0);
    dims[0]=ne; PyArrayObject *lz=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_DOUBLE,0);
    double *s1=calloc((size_t)v,sizeof(double)), *s2=calloc((size_t)v,sizeof(double));
    double *cross=calloc((size_t)ne,sizeof(double)), *me=malloc((size_t)ne*sizeof(double));
    double *row=calloc((size_t)v,sizeof(double)), *col=calloc((size_t)v,sizeof(double));
    pthread_t *threads=NULL; MarginTask *tasks=NULL;
    double *local_y=NULL,*local_1=NULL,*local_2=NULL;
    if(!my||!m1||!m2||!lz||!s1||!s2||!cross||!me||!row||!col){PyErr_NoMemory();goto fail;}
    double *b=PyArray_DATA(base), *q1=PyArray_DATA(r1), *q2=PyArray_DATA(r2);
    double *ep=PyArray_DATA(edge_p), *oy=PyArray_DATA(my), *o1=PyArray_DATA(m1), *o2=PyArray_DATA(m2), *olz=PyArray_DATA(lz);
    int *ea=PyArray_DATA(edge_a), *eb=PyArray_DATA(edge_b), *ay=PyArray_DATA(ya_y), *aa=PyArray_DATA(ya_a), *by=PyArray_DATA(yb_y), *bb=PyArray_DATA(yb_b);
    int *ie=PyArray_DATA(ix_e), *iy=PyArray_DATA(ix_y), *ii1=PyArray_DATA(ix_1), *ii2=PyArray_DATA(ix_2);
    if(workers>1){threads=malloc((size_t)workers*sizeof(*threads));tasks=calloc((size_t)workers,sizeof(*tasks));}
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp i=0;i<n1;i++) s1[aa[i]] += b[ay[i]]*q1[i];
    for(npy_intp i=0;i<n2;i++) s2[bb[i]] += b[by[i]]*q2[i];
    if(workers==1) for(npy_intp i=0;i<ni;i++) cross[ie[i]] += b[iy[i]]*q1[ii1[i]]*q2[ii2[i]];
    else {
        for(int w=0;w<workers;w++){
            npy_intp lo=ni*w/workers, hi=ni*(w+1)/workers;
            if(w>0) while(lo<ni && ie[lo]==ie[lo-1]) lo++;
            if(w<workers-1) while(hi<ni && hi>0 && ie[hi]==ie[hi-1]) hi++;
            tasks[w]=(MarginTask){lo,hi,v,n1,n2,b,q1,q2,me,ie,iy,ii1,ii2,cross,NULL,NULL,NULL,0};
            pthread_create(&threads[w],NULL,margin_worker,&tasks[w]);
        }
        for(int w=0;w<workers;w++) pthread_join(threads[w],NULL);
    }
    double sm=0.0;
    for(npy_intp e=0;e<ne;e++){double z=1+s1[ea[e]]+s2[eb[e]]+cross[e]; olz[e]=log(z); me[e]=ep[e]/z; row[ea[e]]+=me[e]; col[eb[e]]+=me[e]; sm+=me[e];}
    for(npy_intp i=0;i<n1;i++){o1[i]=b[ay[i]]*(1+q1[i])*row[aa[i]]; oy[ay[i]]+=b[ay[i]]*q1[i]*row[aa[i]];}
    for(npy_intp i=0;i<n2;i++){o2[i]=b[by[i]]*(1+q2[i])*col[bb[i]]; oy[by[i]]+=b[by[i]]*q2[i]*col[bb[i]];}
    for(npy_intp y=0;y<v;y++) oy[y]+=b[y]*sm;
    if(workers==1) for(npy_intp i=0;i<ni;i++){double common=b[iy[i]]*me[ie[i]]; o1[ii1[i]]+=common*(1+q1[ii1[i]])*q2[ii2[i]]; o2[ii2[i]]+=common*(1+q2[ii2[i]])*q1[ii1[i]]; oy[iy[i]]+=common*q1[ii1[i]]*q2[ii2[i]];}
    else {
        local_y=calloc((size_t)workers*(size_t)v,sizeof(double));
        local_1=calloc((size_t)workers*(size_t)n1,sizeof(double));
        local_2=calloc((size_t)workers*(size_t)n2,sizeof(double));
        for(int w=0;w<workers;w++){
            tasks[w]=(MarginTask){ni*w/workers,ni*(w+1)/workers,v,n1,n2,b,q1,q2,me,ie,iy,ii1,ii2,NULL,local_y+(size_t)w*v,local_1+(size_t)w*n1,local_2+(size_t)w*n2,1};
            pthread_create(&threads[w],NULL,margin_worker,&tasks[w]);
        }
        for(int w=0;w<workers;w++) pthread_join(threads[w],NULL);
        for(int w=0;w<workers;w++){
            for(npy_intp i=0;i<v;i++) oy[i]+=local_y[(size_t)w*v+i];
            for(npy_intp i=0;i<n1;i++) o1[i]+=local_1[(size_t)w*n1+i];
            for(npy_intp i=0;i<n2;i++) o2[i]+=local_2[(size_t)w*n2+i];
        }
    }
    Py_END_ALLOW_THREADS
    free(threads);free(tasks);free(local_y);free(local_1);free(local_2);
    free(s1);free(s2);free(cross);free(me);free(row);free(col);
    return Py_BuildValue("NNNN",my,m1,m2,lz);
fail:
    free(threads);free(tasks);free(local_y);free(local_1);free(local_2);
    free(s1);free(s2);free(cross);free(me);free(row);free(col);
    Py_XDECREF(my);Py_XDECREF(m1);Py_XDECREF(m2);Py_XDECREF(lz);return NULL;
}

static PyMethodDef methods[]={
    {"fused_margins",fused_margins,METH_VARARGS,"Fused sparse margins."},
    {"intersection_plan",intersection_plan,METH_VARARGS,"Direct compact intersection plan."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_graphical_margin_c",NULL,-1,methods};
PyMODINIT_FUNC PyInit__graphical_margin_c(void){import_array();return PyModule_Create(&module);}
