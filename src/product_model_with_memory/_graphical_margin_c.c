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

static PyObject *layered_intersection_graph(PyObject *self, PyObject *args) {
    PyArrayObject *edge_a,*edge_b,*ya_y,*ya_a,*yb_y,*yb_b;
    PyArrayObject *birth_ya,*birth_yb,*birth_ab;
    int layers;
    if(!PyArg_ParseTuple(args,"O!O!O!O!O!O!O!O!O!i",
        &PyArray_Type,&edge_a,&PyArray_Type,&edge_b,
        &PyArray_Type,&ya_y,&PyArray_Type,&ya_a,
        &PyArray_Type,&yb_y,&PyArray_Type,&yb_b,
        &PyArray_Type,&birth_ya,&PyArray_Type,&birth_yb,
        &PyArray_Type,&birth_ab,&layers))return NULL;
    npy_intp ne=PyArray_SIZE(edge_a),n1=PyArray_SIZE(ya_y),n2=PyArray_SIZE(yb_y),v=0;
    if(layers<1||PyArray_SIZE(edge_b)!=ne||PyArray_SIZE(birth_ab)!=ne||
       PyArray_SIZE(ya_a)!=n1||PyArray_SIZE(birth_ya)!=n1||
       PyArray_SIZE(yb_b)!=n2||PyArray_SIZE(birth_yb)!=n2){
        PyErr_SetString(PyExc_ValueError,"invalid layered graph inputs");return NULL;
    }
    int *ea=PyArray_DATA(edge_a),*eb=PyArray_DATA(edge_b);
    int *ay=PyArray_DATA(ya_y),*aa=PyArray_DATA(ya_a);
    int *by=PyArray_DATA(yb_y),*bb=PyArray_DATA(yb_b);
    npy_uint8 *d1=PyArray_DATA(birth_ya),*d2=PyArray_DATA(birth_yb),*de=PyArray_DATA(birth_ab);
    for(npy_intp i=0;i<ne;i++){if(ea[i]+1>v)v=ea[i]+1;if(eb[i]+1>v)v=eb[i]+1;}
    for(npy_intp i=0;i<n1;i++)if(aa[i]+1>v)v=aa[i]+1;
    for(npy_intp i=0;i<n2;i++)if(bb[i]+1>v)v=bb[i]+1;
    npy_intp *p1=calloc((size_t)v+1,sizeof(*p1)),*p2=calloc((size_t)v+1,sizeof(*p2));
    npy_intp *position=malloc(((size_t)v+1)*sizeof(*position));
    ActivePair *r1=malloc((size_t)n1*sizeof(*r1)),*r2=malloc((size_t)n2*sizeof(*r2));
    npy_intp *counts=calloc((size_t)layers*(size_t)n1,sizeof(*counts));
    npy_intp *cursor=NULL;
    PyObject *ptr_tuple=NULL,*yb_tuple=NULL,*ab_tuple=NULL;
    if(!p1||!p2||!position||(!r1&&n1)||(!r2&&n2)||(!counts&&n1)){
        PyErr_NoMemory();goto fail_layer_graph;
    }
    for(npy_intp i=0;i<n1;i++)p1[aa[i]+1]++;
    for(npy_intp i=0;i<n2;i++)p2[bb[i]+1]++;
    for(npy_intp a=0;a<v;a++){p1[a+1]+=p1[a];p2[a+1]+=p2[a];}
    memcpy(position,p1,((size_t)v+1)*sizeof(*position));
    for(npy_intp i=0;i<n1;i++){npy_intp j=position[aa[i]]++;r1[j]=(ActivePair){ay[i],(int)i};}
    memcpy(position,p2,((size_t)v+1)*sizeof(*position));
    for(npy_intp i=0;i<n2;i++){npy_intp j=position[bb[i]]++;r2[j]=(ActivePair){by[i],(int)i};}
    for(npy_intp a=0;a<v;a++){
        qsort(r1+p1[a],(size_t)(p1[a+1]-p1[a]),sizeof(*r1),compare_active_pair);
        qsort(r2+p2[a],(size_t)(p2[a+1]-p2[a]),sizeof(*r2),compare_active_pair);
    }
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp e=0;e<ne;e++){
        npy_intp i=p1[ea[e]],ie=p1[ea[e]+1],j=p2[eb[e]],je=p2[eb[e]+1];
        while(i<ie&&j<je){
            if(r1[i].y<r2[j].y)i++;else if(r2[j].y<r1[i].y)j++;
            else{
                int depth=d1[r1[i].index];if(d2[r2[j].index]>depth)depth=d2[r2[j].index];if(de[e]>depth)depth=de[e];
                if(depth>=0&&depth<layers)counts[(size_t)depth*n1+r1[i].index]++;
                i++;j++;
            }
        }
    }
    Py_END_ALLOW_THREADS
    ptr_tuple=PyTuple_New(layers);yb_tuple=PyTuple_New(layers);ab_tuple=PyTuple_New(layers);
    cursor=malloc((size_t)layers*(size_t)n1*sizeof(*cursor));
    if(!ptr_tuple||!yb_tuple||!ab_tuple||(!cursor&&n1)){PyErr_NoMemory();goto fail_layer_graph;}
    for(int d=0;d<layers;d++){
        npy_intp ptr_dims[1]={n1+1};
        PyArrayObject *ptr=(PyArrayObject*)PyArray_EMPTY(1,ptr_dims,NPY_INT64,0);
        if(!ptr){goto fail_layer_graph;}
        npy_int64 *raw=PyArray_DATA(ptr);raw[0]=0;
        for(npy_intp i=0;i<n1;i++)raw[i+1]=raw[i]+counts[(size_t)d*n1+i];
        npy_intp edge_dims[1]={raw[n1]};
        PyArrayObject *right=(PyArrayObject*)PyArray_EMPTY(1,edge_dims,NPY_INT32,0);
        PyArrayObject *context=(PyArrayObject*)PyArray_EMPTY(1,edge_dims,NPY_INT32,0);
        if(!right||!context){Py_XDECREF(right);Py_XDECREF(context);Py_DECREF(ptr);goto fail_layer_graph;}
        for(npy_intp i=0;i<n1;i++)cursor[(size_t)d*n1+i]=raw[i];
        PyTuple_SET_ITEM(ptr_tuple,d,(PyObject*)ptr);
        PyTuple_SET_ITEM(yb_tuple,d,(PyObject*)right);
        PyTuple_SET_ITEM(ab_tuple,d,(PyObject*)context);
    }
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp e=0;e<ne;e++){
        npy_intp i=p1[ea[e]],ie=p1[ea[e]+1],j=p2[eb[e]],je=p2[eb[e]+1];
        while(i<ie&&j<je){
            if(r1[i].y<r2[j].y)i++;else if(r2[j].y<r1[i].y)j++;
            else{
                int left=r1[i].index,right=r2[j].index,depth=d1[left];
                if(d2[right]>depth)depth=d2[right];if(de[e]>depth)depth=de[e];
                if(depth>=0&&depth<layers){
                    npy_intp out=cursor[(size_t)depth*n1+left]++;
                    ((int*)PyArray_DATA((PyArrayObject*)PyTuple_GET_ITEM(yb_tuple,depth)))[out]=right;
                    ((int*)PyArray_DATA((PyArrayObject*)PyTuple_GET_ITEM(ab_tuple,depth)))[out]=(int)e;
                }
                i++;j++;
            }
        }
    }
    Py_END_ALLOW_THREADS
    free(p1);free(p2);free(position);free(r1);free(r2);free(counts);free(cursor);
    return Py_BuildValue("NNN",ptr_tuple,yb_tuple,ab_tuple);
fail_layer_graph:
    free(p1);free(p2);free(position);free(r1);free(r2);free(counts);free(cursor);
    Py_XDECREF(ptr_tuple);Py_XDECREF(yb_tuple);Py_XDECREF(ab_tuple);return NULL;
}

static PyObject *ab_major_intersection_graph(PyObject *self, PyObject *args) {
    PyArrayObject *edge_a,*edge_b,*ya_y,*ya_a,*yb_y,*yb_b;
    PyArrayObject *birth_ya,*birth_yb,*birth_ab;
    if(!PyArg_ParseTuple(args,"O!O!O!O!O!O!O!O!O!",
        &PyArray_Type,&edge_a,&PyArray_Type,&edge_b,
        &PyArray_Type,&ya_y,&PyArray_Type,&ya_a,
        &PyArray_Type,&yb_y,&PyArray_Type,&yb_b,
        &PyArray_Type,&birth_ya,&PyArray_Type,&birth_yb,
        &PyArray_Type,&birth_ab))return NULL;
    npy_intp ne=PyArray_SIZE(edge_a),n1=PyArray_SIZE(ya_y),n2=PyArray_SIZE(yb_y),v=0,total=0;
    if(PyArray_SIZE(edge_b)!=ne||PyArray_SIZE(birth_ab)!=ne||
       PyArray_SIZE(ya_a)!=n1||PyArray_SIZE(birth_ya)!=n1||
       PyArray_SIZE(yb_b)!=n2||PyArray_SIZE(birth_yb)!=n2){
        PyErr_SetString(PyExc_ValueError,"invalid AB-major graph inputs");return NULL;
    }
    int *ea=PyArray_DATA(edge_a),*eb=PyArray_DATA(edge_b);
    int *ay=PyArray_DATA(ya_y),*aa=PyArray_DATA(ya_a);
    int *by=PyArray_DATA(yb_y),*bb=PyArray_DATA(yb_b);
    npy_uint8 *d1=PyArray_DATA(birth_ya),*d2=PyArray_DATA(birth_yb),*de=PyArray_DATA(birth_ab);
    for(npy_intp i=0;i<ne;i++){if(ea[i]+1>v)v=ea[i]+1;if(eb[i]+1>v)v=eb[i]+1;}
    for(npy_intp i=0;i<n1;i++)if(aa[i]+1>v)v=aa[i]+1;
    for(npy_intp i=0;i<n2;i++)if(bb[i]+1>v)v=bb[i]+1;
    npy_intp *p1=calloc((size_t)v+1,sizeof(*p1)),*p2=calloc((size_t)v+1,sizeof(*p2));
    npy_intp *position=malloc(((size_t)v+1)*sizeof(*position));
    ActivePair *r1=malloc((size_t)n1*sizeof(*r1)),*r2=malloc((size_t)n2*sizeof(*r2));
    PyArrayObject *ptr=NULL,*o1=NULL,*o2=NULL,*od=NULL;
    if(!p1||!p2||!position||(!r1&&n1)||(!r2&&n2)){PyErr_NoMemory();goto fail_ab_graph;}
    for(npy_intp i=0;i<n1;i++)p1[aa[i]+1]++;
    for(npy_intp i=0;i<n2;i++)p2[bb[i]+1]++;
    for(npy_intp a=0;a<v;a++){p1[a+1]+=p1[a];p2[a+1]+=p2[a];}
    memcpy(position,p1,((size_t)v+1)*sizeof(*position));
    for(npy_intp i=0;i<n1;i++){npy_intp j=position[aa[i]]++;r1[j]=(ActivePair){ay[i],(int)i};}
    memcpy(position,p2,((size_t)v+1)*sizeof(*position));
    for(npy_intp i=0;i<n2;i++){npy_intp j=position[bb[i]]++;r2[j]=(ActivePair){by[i],(int)i};}
    for(npy_intp a=0;a<v;a++){
        qsort(r1+p1[a],(size_t)(p1[a+1]-p1[a]),sizeof(*r1),compare_active_pair);
        qsort(r2+p2[a],(size_t)(p2[a+1]-p2[a]),sizeof(*r2),compare_active_pair);
    }
    npy_intp pdims[1]={ne+1};ptr=(PyArrayObject*)PyArray_ZEROS(1,pdims,NPY_INT64,0);
    if(!ptr){goto fail_ab_graph;}npy_int64 *raw=PyArray_DATA(ptr);
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp e=0;e<ne;e++){
        npy_intp i=p1[ea[e]],ie=p1[ea[e]+1],j=p2[eb[e]],je=p2[eb[e]+1],count=0;
        while(i<ie&&j<je){if(r1[i].y<r2[j].y)i++;else if(r2[j].y<r1[i].y)j++;else{count++;i++;j++;}}
        raw[e+1]=count;
    }
    for(npy_intp e=0;e<ne;e++)raw[e+1]+=raw[e];total=raw[ne];
    Py_END_ALLOW_THREADS
    npy_intp dims[1]={total};
    o1=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_INT32,0);
    o2=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_INT32,0);
    od=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_UINT8,0);
    if(!o1||!o2||!od){PyErr_NoMemory();goto fail_ab_graph;}
    int *out1=PyArray_DATA(o1),*out2=PyArray_DATA(o2);npy_uint8 *outd=PyArray_DATA(od);
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp e=0;e<ne;e++){
        npy_intp i=p1[ea[e]],ie=p1[ea[e]+1],j=p2[eb[e]],je=p2[eb[e]+1],out=raw[e];
        while(i<ie&&j<je){
            if(r1[i].y<r2[j].y)i++;else if(r2[j].y<r1[i].y)j++;
            else{int left=r1[i].index,right=r2[j].index,depth=d1[left];if(d2[right]>depth)depth=d2[right];if(de[e]>depth)depth=de[e];out1[out]=left;out2[out]=right;outd[out]=(npy_uint8)depth;out++;i++;j++;}
        }
    }
    Py_END_ALLOW_THREADS
    free(p1);free(p2);free(position);free(r1);free(r2);
    return Py_BuildValue("NNNN",ptr,o1,o2,od);
fail_ab_graph:
    free(p1);free(p2);free(position);free(r1);free(r2);
    Py_XDECREF(ptr);Py_XDECREF(o1);Py_XDECREF(o2);Py_XDECREF(od);return NULL;
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

typedef struct {
    const npy_int64 *ptr;
    const int *yb;
    const int *ab;
} LayerView;

typedef struct {
    npy_intp row_lo,row_hi,n2,v;
    int layers,phase;
    const LayerView *view;
    const double *base,*q1,*q2,*edge_mass;
    const int *ya_y;
    double *cross,*out_ya,*out_yb,*out_y;
} LayerTask;

static void *layer_worker(void *raw) {
    LayerTask *t=(LayerTask*)raw;
    if(t->phase==0){
        for(int d=0;d<t->layers;d++) for(npy_intp i=t->row_lo;i<t->row_hi;i++){
            double left=t->base[t->ya_y[i]]*t->q1[i];
            for(npy_int64 p=t->view[d].ptr[i];p<t->view[d].ptr[i+1];p++)
                t->cross[t->view[d].ab[p]] += left*t->q2[t->view[d].yb[p]];
        }
    }else{
        for(int d=0;d<t->layers;d++) for(npy_intp i=t->row_lo;i<t->row_hi;i++){
            int y=t->ya_y[i];
            for(npy_int64 p=t->view[d].ptr[i];p<t->view[d].ptr[i+1];p++){
                int j=t->view[d].yb[p],e=t->view[d].ab[p];
                double common=t->base[y]*t->edge_mass[e];
                t->out_ya[i]+=common*(1+t->q1[i])*t->q2[j];
                t->out_yb[j]+=common*(1+t->q2[j])*t->q1[i];
                t->out_y[y]+=common*t->q1[i]*t->q2[j];
            }
        }
    }
    return NULL;
}

static PyObject *fused_margins_layered(PyObject *self, PyObject *args) {
    PyArrayObject *base, *r1, *r2, *edge_a, *edge_b, *edge_p;
    PyArrayObject *ya_y, *ya_a, *yb_y, *yb_b;
    PyObject *row_layers, *yb_layers, *ab_layers;
    int checkpoint,workers=1;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!O!i|i",
            &PyArray_Type,&base, &PyArray_Type,&r1, &PyArray_Type,&r2,
            &PyArray_Type,&edge_a, &PyArray_Type,&edge_b,
            &PyArray_Type,&edge_p, &PyArray_Type,&ya_y,
            &PyArray_Type,&ya_a, &PyArray_Type,&yb_y,
            &PyArray_Type,&yb_b, &PyTuple_Type,&row_layers,
            &PyTuple_Type,&yb_layers, &PyTuple_Type,&ab_layers,
            &checkpoint,&workers)) return NULL;
    if(workers<1)workers=1;
    Py_ssize_t layer_count=PyTuple_GET_SIZE(row_layers);
    if(layer_count<1 || PyTuple_GET_SIZE(yb_layers)!=layer_count ||
       PyTuple_GET_SIZE(ab_layers)!=layer_count || checkpoint<0 ||
       checkpoint>=layer_count){
        PyErr_SetString(PyExc_ValueError,"invalid layered intersection graph");
        return NULL;
    }
    npy_intp v=PyArray_SIZE(base), n1=PyArray_SIZE(r1), n2=PyArray_SIZE(r2);
    npy_intp ne=PyArray_SIZE(edge_p), dims[1];
    dims[0]=v; PyArrayObject *my=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_DOUBLE,0);
    dims[0]=n1; PyArrayObject *m1=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_DOUBLE,0);
    dims[0]=n2; PyArrayObject *m2=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_DOUBLE,0);
    dims[0]=ne; PyArrayObject *lz=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_DOUBLE,0);
    PyArrayObject *unstable=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_UINT8,0);
    double *s1=calloc((size_t)v,sizeof(double)), *s2=calloc((size_t)v,sizeof(double));
    double *cs1=calloc((size_t)v,sizeof(double)), *cs2=calloc((size_t)v,sizeof(double));
    double *cross=calloc((size_t)ne,sizeof(double)), *ccross=calloc((size_t)ne,sizeof(double)), *me=malloc((size_t)ne*sizeof(double));
    double *row=calloc((size_t)v,sizeof(double)), *col=calloc((size_t)v,sizeof(double));
    LayerView *view=NULL;pthread_t *threads=NULL;LayerTask *tasks=NULL;
    double *local_cross=NULL,*local_yb=NULL,*local_y=NULL;
    if(!my||!m1||!m2||!lz||!unstable||!s1||!s2||!cs1||!cs2||!cross||!ccross||!me||!row||!col){PyErr_NoMemory();goto fail_layered;}
    double *b=PyArray_DATA(base), *q1=PyArray_DATA(r1), *q2=PyArray_DATA(r2);
    double *ep=PyArray_DATA(edge_p), *oy=PyArray_DATA(my), *o1=PyArray_DATA(m1), *o2=PyArray_DATA(m2), *olz=PyArray_DATA(lz);
    npy_uint8 *bad=PyArray_DATA(unstable);
    int *ea=PyArray_DATA(edge_a), *eb=PyArray_DATA(edge_b), *ay=PyArray_DATA(ya_y), *aa=PyArray_DATA(ya_a), *by=PyArray_DATA(yb_y), *bb=PyArray_DATA(yb_b);
    /* Validate layer array sizes while the GIL is held. */
    view=malloc((size_t)(checkpoint+1)*sizeof(*view));
    if(!view){PyErr_NoMemory();goto fail_layered;}
    for(int d=0;d<=checkpoint;d++){
        PyObject *rp_obj=PyTuple_GET_ITEM(row_layers,d);
        PyObject *yb_obj=PyTuple_GET_ITEM(yb_layers,d);
        PyObject *ab_obj=PyTuple_GET_ITEM(ab_layers,d);
        if(!PyArray_Check(rp_obj)||!PyArray_Check(yb_obj)||!PyArray_Check(ab_obj)||
           PyArray_SIZE((PyArrayObject*)rp_obj)<n1+1 ||
           PyArray_SIZE((PyArrayObject*)yb_obj)!=PyArray_SIZE((PyArrayObject*)ab_obj)){
            PyErr_SetString(PyExc_ValueError,"invalid layered CSR arrays");
            goto fail_layered;
        }
        view[d]=(LayerView){
            PyArray_DATA((PyArrayObject*)rp_obj),
            PyArray_DATA((PyArrayObject*)yb_obj),
            PyArray_DATA((PyArrayObject*)ab_obj)
        };
    }
    if(workers>1){
        threads=malloc((size_t)workers*sizeof(*threads));
        tasks=calloc((size_t)workers,sizeof(*tasks));
        local_cross=calloc((size_t)workers*(size_t)ne,sizeof(double));
        local_yb=calloc((size_t)workers*(size_t)n2,sizeof(double));
        local_y=calloc((size_t)workers*(size_t)v,sizeof(double));
        if(!threads||!tasks||!local_cross||!local_yb||!local_y){
            PyErr_NoMemory();goto fail_layered;
        }
    }
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp i=0;i<n1;i++){
        int a=aa[i];double value=b[ay[i]]*q1[i]-cs1[a];
        double total=s1[a]+value;cs1[a]=(total-s1[a])-value;s1[a]=total;
    }
    for(npy_intp i=0;i<n2;i++){
        int a=bb[i];double value=b[by[i]]*q2[i]-cs2[a];
        double total=s2[a]+value;cs2[a]=(total-s2[a])-value;s2[a]=total;
    }
    if(workers==1){
        for(int d=0;d<=checkpoint;d++) for(npy_intp i=0;i<n1;i++){
            double left=b[ay[i]]*q1[i];
            for(npy_int64 p=view[d].ptr[i];p<view[d].ptr[i+1];p++){
                int e=view[d].ab[p];double value=left*q2[view[d].yb[p]]-ccross[e];
                double total=cross[e]+value;ccross[e]=(total-cross[e])-value;cross[e]=total;
            }
        }
    }else{
        for(int w=0;w<workers;w++){
            tasks[w]=(LayerTask){n1*w/workers,n1*(w+1)/workers,n2,v,
                checkpoint+1,0,view,b,q1,q2,NULL,ay,
                local_cross+(size_t)w*ne,NULL,NULL,NULL};
            pthread_create(&threads[w],NULL,layer_worker,&tasks[w]);
        }
        for(int w=0;w<workers;w++)pthread_join(threads[w],NULL);
        for(npy_intp e=0;e<ne;e++)for(int w=0;w<workers;w++){
            double value=local_cross[(size_t)w*ne+e]-ccross[e];
            double total=cross[e]+value;ccross[e]=(total-cross[e])-value;cross[e]=total;
        }
        free(local_cross);local_cross=NULL;
    }
    double sm=0.0;
    for(npy_intp e=0;e<ne;e++){
        double z=1+s1[ea[e]]+s2[eb[e]]+cross[e];
        double scale=1+fabs(s1[ea[e]])+fabs(s2[eb[e]])+fabs(cross[e]);
        if(!isfinite(z)||z<=0||scale>1e10*fmax(fabs(z),1e-300)){
            bad[e]=1;olz[e]=NAN;me[e]=0;
        }else{
            olz[e]=log(z);me[e]=ep[e]/z;
            row[ea[e]]+=me[e];col[eb[e]]+=me[e];sm+=me[e];
        }
    }
    for(npy_intp i=0;i<n1;i++){o1[i]=b[ay[i]]*(1+q1[i])*row[aa[i]];oy[ay[i]]+=b[ay[i]]*q1[i]*row[aa[i]];}
    for(npy_intp i=0;i<n2;i++){o2[i]=b[by[i]]*(1+q2[i])*col[bb[i]];oy[by[i]]+=b[by[i]]*q2[i]*col[bb[i]];}
    for(npy_intp y=0;y<v;y++) oy[y]+=b[y]*sm;
    if(workers==1){
        for(int d=0;d<=checkpoint;d++) for(npy_intp i=0;i<n1;i++)
        for(npy_int64 p=view[d].ptr[i];p<view[d].ptr[i+1];p++){
            int j=view[d].yb[p],e=view[d].ab[p],y=ay[i];double common=b[y]*me[e];
            o1[i]+=common*(1+q1[i])*q2[j];o2[j]+=common*(1+q2[j])*q1[i];oy[y]+=common*q1[i]*q2[j];
        }
    }else{
        for(int w=0;w<workers;w++){
            tasks[w]=(LayerTask){n1*w/workers,n1*(w+1)/workers,n2,v,
                checkpoint+1,1,view,b,q1,q2,me,ay,NULL,o1,
                local_yb+(size_t)w*n2,local_y+(size_t)w*v};
            pthread_create(&threads[w],NULL,layer_worker,&tasks[w]);
        }
        for(int w=0;w<workers;w++)pthread_join(threads[w],NULL);
        for(int w=0;w<workers;w++){
            for(npy_intp j=0;j<n2;j++)o2[j]+=local_yb[(size_t)w*n2+j];
            for(npy_intp y=0;y<v;y++)oy[y]+=local_y[(size_t)w*v+y];
        }
    }
    Py_END_ALLOW_THREADS
    free(view);free(threads);free(tasks);free(local_cross);free(local_yb);free(local_y);
    free(s1);free(s2);free(cs1);free(cs2);free(cross);free(ccross);free(me);free(row);free(col);
    return Py_BuildValue("NNNNN",my,m1,m2,lz,unstable);
fail_layered:
    free(view);free(threads);free(tasks);free(local_cross);free(local_yb);free(local_y);
    free(s1);free(s2);free(cs1);free(cs2);free(cross);free(ccross);free(me);free(row);free(col);
    Py_XDECREF(my);Py_XDECREF(m1);Py_XDECREF(m2);Py_XDECREF(lz);Py_XDECREF(unstable);return NULL;
}

typedef struct {
    npy_intp lo,hi,v,n1,n2;long long offset;int checkpoint,phase;
    const double *b,*q1,*q2,*ep,*s1,*s2,*me;
    const int *ea,*eb,*ay,*aa,*by,*bb,*first,*second;
    const npy_int64 *ptr;const npy_uint8 *depth;
    double *olz,*out_y,*out1,*out2,*row,*col,sm;
    npy_uint8 *bad;
    const int *global_edge;
} ABMarginTask;

static void *ab_margin_worker(void *raw){
    ABMarginTask *t=(ABMarginTask*)raw;
    if(t->phase==0){
        t->sm=0.0;
        for(npy_intp e=t->lo;e<t->hi;e++){
            npy_intp ge=t->global_edge?t->global_edge[e]:(npy_intp)t->offset+e;double cross=0.0,cc=0.0;
            for(npy_int64 p=t->ptr[ge];p<t->ptr[ge+1];p++)if(t->depth[p]<=t->checkpoint){int i=t->first[p],j=t->second[p];double value=t->b[t->ay[i]]*t->q1[i]*t->q2[j]-cc,total=cross+value;cc=(total-cross)-value;cross=total;}
            double z=1+t->s1[t->ea[e]]+t->s2[t->eb[e]]+cross,scale=1+fabs(t->s1[t->ea[e]])+fabs(t->s2[t->eb[e]])+fabs(cross);
            if(!isfinite(z)||z<=0||scale>1e10*fmax(fabs(z),1e-300)){t->bad[e]=1;t->olz[e]=NAN;((double*)t->me)[e]=0;}
            else{t->olz[e]=log(z);((double*)t->me)[e]=t->ep[e]/z;t->row[t->ea[e]]+=t->me[e];t->col[t->eb[e]]+=t->me[e];t->sm+=t->me[e];}
        }
    }else{
        for(npy_intp e=t->lo;e<t->hi;e++)if(t->me[e]!=0){npy_intp ge=t->global_edge?t->global_edge[e]:(npy_intp)t->offset+e;for(npy_int64 p=t->ptr[ge];p<t->ptr[ge+1];p++)if(t->depth[p]<=t->checkpoint){int i=t->first[p],j=t->second[p],y=t->ay[i];double common=t->b[y]*t->me[e];t->out1[i]+=common*(1+t->q1[i])*t->q2[j];t->out2[j]+=common*(1+t->q2[j])*t->q1[i];t->out_y[y]+=common*t->q1[i]*t->q2[j];}}
    }
    return NULL;
}

static npy_intp ab_edge_cut(const npy_int64 *ptr,npy_intp offset,npy_intp ne,int part,int parts){
    if(part<=0)return 0;if(part>=parts)return ne;
    npy_int64 start=ptr[offset],total=ptr[offset+ne]-start;
    npy_int64 target=start+(npy_int64)((long double)total*part/parts);
    npy_intp lo=0,hi=ne;
    while(lo<hi){npy_intp mid=lo+(hi-lo)/2;if(ptr[offset+mid]<target)lo=mid+1;else hi=mid;}
    return lo;
}

static PyObject *fused_margins_ab_major(PyObject *self, PyObject *args) {
    PyArrayObject *base,*r1,*r2,*edge_a,*edge_b,*edge_p,*ya_y,*ya_a,*yb_y,*yb_b;
    PyArrayObject *edge_ptr,*ix1,*ix2,*birth,*edge_ids=NULL;
    int checkpoint,workers=1;long long edge_offset;
    if(!PyArg_ParseTuple(args,"O!O!O!O!O!O!O!O!O!O!O!O!O!O!iL|iO!",
        &PyArray_Type,&base,&PyArray_Type,&r1,&PyArray_Type,&r2,
        &PyArray_Type,&edge_a,&PyArray_Type,&edge_b,&PyArray_Type,&edge_p,
        &PyArray_Type,&ya_y,&PyArray_Type,&ya_a,&PyArray_Type,&yb_y,&PyArray_Type,&yb_b,
        &PyArray_Type,&edge_ptr,&PyArray_Type,&ix1,&PyArray_Type,&ix2,&PyArray_Type,&birth,
        &checkpoint,&edge_offset,&workers,&PyArray_Type,&edge_ids))return NULL;
    if(workers<1)workers=1;
    npy_intp v=PyArray_SIZE(base),n1=PyArray_SIZE(r1),n2=PyArray_SIZE(r2),ne=PyArray_SIZE(edge_p);
    if(edge_offset<0||(!edge_ids&&edge_offset+ne+1>PyArray_SIZE(edge_ptr))||
       PyArray_SIZE(edge_a)!=ne||PyArray_SIZE(edge_b)!=ne||
       (edge_ids&&PyArray_SIZE(edge_ids)!=ne)||
       PyArray_SIZE(ix1)!=PyArray_SIZE(ix2)||PyArray_SIZE(ix1)!=PyArray_SIZE(birth)){
        PyErr_SetString(PyExc_ValueError,"invalid AB-major margin inputs");return NULL;
    }
    npy_intp dims[1];dims[0]=v;PyArrayObject *my=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_DOUBLE,0);
    dims[0]=n1;PyArrayObject *m1=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_DOUBLE,0);
    dims[0]=n2;PyArrayObject *m2=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_DOUBLE,0);
    dims[0]=ne;PyArrayObject *lz=(PyArrayObject*)PyArray_EMPTY(1,dims,NPY_DOUBLE,0);
    PyArrayObject *unstable=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_UINT8,0);
    double *s1=calloc((size_t)v,sizeof(double)),*s2=calloc((size_t)v,sizeof(double));
    double *cs1=calloc((size_t)v,sizeof(double)),*cs2=calloc((size_t)v,sizeof(double));
    double *me=malloc((size_t)ne*sizeof(double)),*row=calloc((size_t)v,sizeof(double)),*col=calloc((size_t)v,sizeof(double));
    pthread_t *threads=NULL;ABMarginTask *tasks=NULL;double *local_row=NULL,*local_col=NULL,*local_y=NULL,*local_1=NULL,*local_2=NULL;
    if(!my||!m1||!m2||!lz||!unstable||!s1||!s2||!cs1||!cs2||!me||!row||!col){PyErr_NoMemory();goto fail_ab_margin;}
    double *b=PyArray_DATA(base),*q1=PyArray_DATA(r1),*q2=PyArray_DATA(r2),*ep=PyArray_DATA(edge_p);
    double *oy=PyArray_DATA(my),*o1=PyArray_DATA(m1),*o2=PyArray_DATA(m2),*olz=PyArray_DATA(lz);
    int *ea=PyArray_DATA(edge_a),*eb=PyArray_DATA(edge_b),*ay=PyArray_DATA(ya_y),*aa=PyArray_DATA(ya_a),*by=PyArray_DATA(yb_y),*bb=PyArray_DATA(yb_b);
    npy_int64 *ptr=PyArray_DATA(edge_ptr);int *first=PyArray_DATA(ix1),*second=PyArray_DATA(ix2),*global=edge_ids?PyArray_DATA(edge_ids):NULL;npy_uint8 *depth=PyArray_DATA(birth),*bad=PyArray_DATA(unstable);
    if(global)for(npy_intp e=0;e<ne;e++)if(global[e]<0||global[e]+1>=PyArray_SIZE(edge_ptr)){PyErr_SetString(PyExc_ValueError,"indexed AB edge outside graph");goto fail_ab_margin;}
    if(workers>1){
        threads=malloc((size_t)workers*sizeof(*threads));tasks=calloc((size_t)workers,sizeof(*tasks));
        local_row=calloc((size_t)workers*v,sizeof(double));local_col=calloc((size_t)workers*v,sizeof(double));
        local_y=calloc((size_t)workers*v,sizeof(double));local_1=calloc((size_t)workers*n1,sizeof(double));local_2=calloc((size_t)workers*n2,sizeof(double));
        if(!threads||!tasks||!local_row||!local_col||!local_y||!local_1||!local_2){PyErr_NoMemory();goto fail_ab_margin;}
    }
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp i=0;i<n1;i++){int a=aa[i];double value=b[ay[i]]*q1[i]-cs1[a],total=s1[a]+value;cs1[a]=(total-s1[a])-value;s1[a]=total;}
    for(npy_intp i=0;i<n2;i++){int a=bb[i];double value=b[by[i]]*q2[i]-cs2[a],total=s2[a]+value;cs2[a]=(total-s2[a])-value;s2[a]=total;}
    double sm=0.0;
    if(workers==1){ABMarginTask task={0,ne,v,n1,n2,edge_offset,checkpoint,0,b,q1,q2,ep,s1,s2,me,ea,eb,ay,aa,by,bb,first,second,ptr,depth,olz,NULL,NULL,NULL,row,col,0,bad,global};ab_margin_worker(&task);sm=task.sm;}
    else{
        for(int w=0;w<workers;w++){npy_intp lo=global?ne*w/workers:ab_edge_cut(ptr,(npy_intp)edge_offset,ne,w,workers),hi=global?ne*(w+1)/workers:ab_edge_cut(ptr,(npy_intp)edge_offset,ne,w+1,workers);tasks[w]=(ABMarginTask){lo,hi,v,n1,n2,edge_offset,checkpoint,0,b,q1,q2,ep,s1,s2,me,ea,eb,ay,aa,by,bb,first,second,ptr,depth,olz,NULL,NULL,NULL,local_row+(size_t)w*v,local_col+(size_t)w*v,0,bad,global};pthread_create(&threads[w],NULL,ab_margin_worker,&tasks[w]);}
        for(int w=0;w<workers;w++)pthread_join(threads[w],NULL);
        for(int w=0;w<workers;w++){sm+=tasks[w].sm;for(npy_intp x=0;x<v;x++){row[x]+=local_row[(size_t)w*v+x];col[x]+=local_col[(size_t)w*v+x];}}
        free(local_row);local_row=NULL;free(local_col);local_col=NULL;
    }
    for(npy_intp i=0;i<n1;i++){o1[i]=b[ay[i]]*(1+q1[i])*row[aa[i]];oy[ay[i]]+=b[ay[i]]*q1[i]*row[aa[i]];}
    for(npy_intp i=0;i<n2;i++){o2[i]=b[by[i]]*(1+q2[i])*col[bb[i]];oy[by[i]]+=b[by[i]]*q2[i]*col[bb[i]];}
    for(npy_intp y=0;y<v;y++)oy[y]+=b[y]*sm;
    if(workers==1){ABMarginTask task={0,ne,v,n1,n2,edge_offset,checkpoint,1,b,q1,q2,ep,s1,s2,me,ea,eb,ay,aa,by,bb,first,second,ptr,depth,olz,oy,o1,o2,NULL,NULL,0,bad,global};ab_margin_worker(&task);}
    else{
        for(int w=0;w<workers;w++){npy_intp lo=global?ne*w/workers:ab_edge_cut(ptr,(npy_intp)edge_offset,ne,w,workers),hi=global?ne*(w+1)/workers:ab_edge_cut(ptr,(npy_intp)edge_offset,ne,w+1,workers);tasks[w]=(ABMarginTask){lo,hi,v,n1,n2,edge_offset,checkpoint,1,b,q1,q2,ep,s1,s2,me,ea,eb,ay,aa,by,bb,first,second,ptr,depth,olz,local_y+(size_t)w*v,local_1+(size_t)w*n1,local_2+(size_t)w*n2,NULL,NULL,0,bad,global};pthread_create(&threads[w],NULL,ab_margin_worker,&tasks[w]);}
        for(int w=0;w<workers;w++)pthread_join(threads[w],NULL);
        for(int w=0;w<workers;w++){for(npy_intp y=0;y<v;y++)oy[y]+=local_y[(size_t)w*v+y];for(npy_intp i=0;i<n1;i++)o1[i]+=local_1[(size_t)w*n1+i];for(npy_intp j=0;j<n2;j++)o2[j]+=local_2[(size_t)w*n2+j];}
    }
    Py_END_ALLOW_THREADS
    free(threads);free(tasks);free(local_y);free(local_1);free(local_2);free(s1);free(s2);free(cs1);free(cs2);free(me);free(row);free(col);
    return Py_BuildValue("NNNNN",my,m1,m2,lz,unstable);
fail_ab_margin:
    free(threads);free(tasks);free(local_row);free(local_col);free(local_y);free(local_1);free(local_2);free(s1);free(s2);free(cs1);free(cs2);free(me);free(row);free(col);
    Py_XDECREF(my);Py_XDECREF(m1);Py_XDECREF(m2);Py_XDECREF(lz);Py_XDECREF(unstable);return NULL;
}

static PyMethodDef methods[]={
    {"ab_major_intersection_graph",ab_major_intersection_graph,METH_VARARGS,"Direct birth-tagged AB-major graph."},
    {"fused_margins_ab_major",fused_margins_ab_major,METH_VARARGS,"Fused AB-major block margins."},
    {"fused_margins",fused_margins,METH_VARARGS,"Fused sparse margins."},
    {"fused_margins_layered",fused_margins_layered,METH_VARARGS,"Fused birth-layered CSR margins."},
    {"intersection_plan",intersection_plan,METH_VARARGS,"Direct compact intersection plan."},
    {"layered_intersection_graph",layered_intersection_graph,METH_VARARGS,"Direct birth-layered CSR graph."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_graphical_margin_c",NULL,-1,methods};
PyMODINIT_FUNC PyInit__graphical_margin_c(void){import_array();return PyModule_Create(&module);}
