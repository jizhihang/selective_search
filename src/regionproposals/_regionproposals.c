//#undef NDEBUG
//#define DEBUG 1
#define NDEBUG 1
#include <assert.h>

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#define THRESHOLD(size, c) (c/size)

typedef struct {
    int top;
    int bottom;
    int left;
    int right;
} region;

typedef struct {
    int rank;
    int p;
    int size;
} uni_elt;

typedef struct {
    uni_elt * elts;
    int num;
} universe;

typedef struct {
    float w;
    int a, b;
} edge;

inline float minf (float a, float b) { return (a < b) ? a : b; }
inline float maxf (float a, float b) { return (a > b) ? a : b; }
inline int min (int a, int b) { return (a < b) ? a : b; }
inline int max (int a, int b) { return (a > b) ? a : b; }

static universe * universe_create (int elements) {
    universe * this = (universe*)malloc(sizeof(universe));
    this->elts = (uni_elt*)malloc(sizeof(uni_elt)*elements);
    this->num = elements;
    int i;
    for (i = 0; i < elements; ++i) {
        this->elts[i].rank = 0;
        this->elts[i].size = 1;
        this->elts[i].p = i;
    }
    return this;
}

static void universe_destroy (universe * this) {
    free(this->elts);
    free(this);
}

static int universe_size (universe * this, int x) {
    return this->elts[x].size;
}

static int universe_num_sets (universe * this) {
    return this->num;
}

static int universe_find (universe * this, int x) {
    int y = x;
    while (y != this->elts[y].p) {
        y = this->elts[y].p;
    }
    this->elts[x].p = y;
    return y;
}

static void universe_join (universe * this, int x, int y) {
    if (this->elts[x].rank > this->elts[y].rank) {
        this->elts[y].p = x;
        this->elts[x].size += this->elts[y].size;
    } else {
        this->elts[x].p = y;
        this->elts[y].size += this->elts[x].size;
        if (this->elts[x].rank == this->elts[y].rank) {
            this->elts[y].rank++;
        }
    }
    this->num--;
}

static inline float color_similarity (float * hist1, float * hist2) {
    float sim = 0;
    int i;
    for (i = 0; i < 75; ++i) {
        assert(hist1[i] >= 0);
        assert(hist2[i] >= 0);
        assert(hist1[i] <= 1);
        assert(hist2[i] <= 1);
        sim += minf(hist1[i], hist2[i]);
    }
    return sim;
}

static inline float size_similarity (int a, int b, int size) {
    return 1.0 - (a + b)/size;
}

static inline float fill_similarity (region * ra, region * rb, int a, int b, int size) {
    int width = max(ra->right, rb->right) - min(ra->left, rb->left);
    int height = max(ra->bottom, rb->bottom) - min(ra->top, rb->top);
    assert (width >= 0);
    assert (height >= 0);
    return 1.0 - (width*height - a - b)/size;
}

static inline float square(float x) { return x*x; };

static inline float diff(PyArrayObject * image,
             int x1, int y1, int x2, int y2) {
    float r1 = *(float*)PyArray_GETPTR3(image, y1, x1, 0);
    float g1 = *(float*)PyArray_GETPTR3(image, y1, x1, 1);
    float b1 = *(float*)PyArray_GETPTR3(image, y1, x1, 2);
    float r2 = *(float*)PyArray_GETPTR3(image, y2, x2, 0);
    float g2 = *(float*)PyArray_GETPTR3(image, y2, x2, 1);
    float b2 = *(float*)PyArray_GETPTR3(image, y2, x2, 2);
    float r_diff = (float)r1 - (float)r2;
    float g_diff = (float)g1 - (float)g2;
    float b_diff = (float)b1 - (float)b2;
    return sqrt(square(r_diff) + square(g_diff) + square(b_diff));
}

int comp (const void * elem1, const void * elem2) 
{
    edge f = *((edge*)elem1);
    edge s = *((edge*)elem2);
    if (f.w > s.w) return  1;
    if (f.w < s.w) return -1;
    return 0;
}

static universe *segment_graph(int num_vertices, int num_edges, edge *edges, float c) {
    qsort (edges, num_edges, sizeof(edge), comp);

    universe *u = universe_create (num_vertices);

    float *threshold = (float*)malloc(sizeof(float)*num_vertices);
    int i;
    for (i = 0; i < num_vertices; i++)
        threshold[i] = THRESHOLD(1,c);

    for (i = 0; i < num_edges; i++) {
        edge *pedge = edges + i;

        int a = universe_find (u, pedge->a);
        int b = universe_find (u, pedge->b);
        if (a != b) {
            if ((pedge->w <= threshold[a]) && (pedge->w <= threshold[b])) {
                universe_join (u, a, b);
                a = universe_find (u, a);
                threshold[a] = pedge->w + THRESHOLD(universe_size (u, a), c);
            }
        }
    }

    free (threshold);
    return u;
}

static PyObject * segment(PyObject * self, PyObject * args, PyObject * kwargs)
{
    PyObject * pyimage;
    float c;
    int min_size, range1, range2, range3;
    static char * keywords[] = {"image", "c", "min_size", "range1", "range2", "range3", NULL};
    if (!PyArg_ParseTupleAndKeywords (args, kwargs, "O!fiiii", keywords, &PyArray_Type, &pyimage, &c, &min_size, &range1, &range2, &range3)) {
        return NULL;
    }

    PyArrayObject * image = (PyArrayObject*)pyimage;

    if (PyArray_NDIM(image) != 3) {
        PyErr_SetString(PyExc_TypeError, "3 channel image required");
        return NULL;
    }

    if (PyArray_TYPE(image) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "byte image required");
        return NULL;
    }

    npy_intp * dims = PyArray_DIMS(image);
    npy_intp height = dims[0];
    npy_intp width = dims[1];

    if (dims[2] != 3) {
        PyErr_SetString(PyExc_TypeError, "image must be a 3 channel image");
        return NULL;
    }

    edge *edges = (edge*)malloc(sizeof(edge)*width*height*4);
    int num = 0;
    int y;
    int x;
    int i, j;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (x < width-1) {
                edges[num].a = y * width + x;
                edges[num].b = y * width + (x+1);
                edges[num].w = diff(image, x, y, x+1, y);
                num++;
            }

            if (y < height-1) {
                edges[num].a = y * width + x;
                edges[num].b = (y+1) * width + x;
                edges[num].w = diff(image, x, y, x, y+1);
                num++;
            }

            if ((x < width-1) && (y < height-1)) {
                edges[num].a = y * width + x;
                edges[num].b = (y+1) * width + (x+1);
                edges[num].w = diff(image, x, y, x+1, y+1);
                num++;
            }

            if ((x < width-1) && (y > 0)) {
                edges[num].a = y * width + x;
                edges[num].b = (y-1) * width + (x+1);
                edges[num].w = diff(image, x, y, x+1, y-1);
                num++;
            }
        }
    }

    universe *u = segment_graph(width*height, num, edges, c);
    
    for (i = 0; i < num; i++) {
        int a = universe_find(u, edges[i].a);
        int b = universe_find(u, edges[i].b);
        if ((a != b) && ((universe_size (u, a) < min_size) || (universe_size (u, b) < min_size)))
            universe_join (u, a, b);
    }
    free(edges);
    int num_ccs = universe_num_sets (u);

    npy_intp out_dims [] = {height, width, 3};
    PyObject * pyoutput = PyArray_SimpleNew (3, out_dims, NPY_UINT8);
    PyArrayObject * output = (PyArrayObject*)pyoutput;

    uint8_t * colors = (uint8_t*)malloc(sizeof(uint8_t)*width*height*3);
    for (i = 0; i < width*height*3; i++) {
        colors[i] = rand()/(RAND_MAX + 1.0) * 255;
    }
    
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            int comp = universe_find (u, y * width + x);
            for (i = 0; i < 3; ++i) {
                *(uint8_t*)PyArray_GETPTR3(output, y, x, i) = colors[3*comp + i];
            }
        }
    }    

    free(colors);

    region * regions = (region*)malloc(sizeof(region)*num_ccs);
    for (i = 0; i < num_ccs; ++i) {
        regions[i].top = height;
        regions[i].bottom = 0;
        regions[i].left = width;
        regions[i].right = 0;
    }

    float * histogram = (float*)malloc(sizeof(float)*75*num_ccs);
    memset(histogram, 0, sizeof(float)*75*num_ccs);

    int * components = (int*)malloc(sizeof(int)*num_ccs);
    int next_component = 0;

    int * component_map = (int*)malloc(sizeof(int)*height*width);

    int * counts = (int*)malloc(sizeof(int)*num_ccs);
    memset(counts, 0, sizeof(int)*num_ccs);

    const float n1 = range1 / 25.0;
    const float n2 = range2 / 25.0;
    const float n3 = range3 / 25.0;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            int comp = universe_find (u, y * width + x);
            int component_id = -1;
            for (i = 0; i < next_component; ++i) {
                if (components[i] == comp) {
                    component_id = i;
                    break;
                }
            }
            if (i == next_component) {
                components[next_component] = comp;
                component_id = next_component;
                ++next_component;
            }
            component_map[y * width + x] = component_id;
            region * r = regions + component_id;
            r->top = min(r->top, y);
            r->bottom = max(r->bottom, y);
            r->left = min(r->left, x);
            r->right = max(r->right, x);
            float c1 = *(float*)PyArray_GETPTR3(image, y, x, 0);
            float c2 = *(float*)PyArray_GETPTR3(image, y, x, 1);
            float c3 = *(float*)PyArray_GETPTR3(image, y, x, 2);
            int r_bin = c1/n1;
            int g_bin = c2/n2;
            int b_bin = c3/n3;
            assert(histogram[75*component_id + r_bin] >= 0);
            assert(histogram[75*component_id + 25 + g_bin] >= 0);
            assert(histogram[75*component_id + 50 + b_bin] >= 0);
            histogram[75*component_id + r_bin]++;
            histogram[75*component_id + 25 + g_bin]++;
            histogram[75*component_id + 50 + b_bin]++;
            counts[component_id]++;
        }
    }
    assert (next_component == num_ccs);

    PyObject * region_list = PyList_New(0);
    /*
    for (i = 0; i < num_ccs; ++i) {
        PyList_SetItem (region_list, i, Py_BuildValue("iiiii", regions[i].left, regions[i].right, regions[i].top, regions[i].bottom, 0));
    }
    */

    for (i = 0; i < num_ccs; ++i) {
        float max_val = 0;
        int j;
        for (j = 0; j < 75; ++j) {
            assert(histogram[75*i + j] >= 0);
            max_val = max(max_val, histogram[75*i + j]);
        }
        for (j = 0; j < 75; ++j) {
            histogram[75*i + j] /= max_val;
            assert(histogram[75*i + j] >= 0);
        }
    }

    uint8_t * adjacency = (uint8_t*)malloc(sizeof(uint8_t)*num_ccs*num_ccs);
    memset(adjacency, 0, sizeof(uint8_t)*num_ccs*num_ccs);
    for (y = 0; y < height-1; ++y) {
        for (x = 0; x < width-1; ++x) {
            int component1 = component_map[y * width + x];
            int component2 = component_map[y * width + x + 1];
            int component3 = component_map[y * width + x + width];

            if (component1 != component2) {
                adjacency[component1 * num_ccs + component2] = 1;
                adjacency[component2 * num_ccs + component1] = 1;
            }

            if (component1 != component3) {
                adjacency[component1 * num_ccs + component3] = 1;
                adjacency[component3 * num_ccs + component1] = 1;
            }
        }
    }

    assert (adjacency[188 * num_ccs + 410] == 0);

    float a1 = 1;
    float a2 = 1;
    float a3 = 1;

    float * similarity_table = (float*)malloc(sizeof(float)*num_ccs*num_ccs);
    int size = height * width;
    for (i = 0; i < num_ccs; ++i) {
        for (j = i + 1; j < num_ccs; ++j) {
            float color_sim = color_similarity (histogram + 75 * i, histogram + 75 * j);
            float size_sim = size_similarity (counts[i], counts[j], size);
            float fill_sim = fill_similarity (regions + i, regions + j, counts[i], counts[j], size);
            float similarity = a1 * color_sim + a2 * size_sim + a3 * fill_sim;
            similarity_table[i * num_ccs + j] = similarity;
            similarity_table[j * num_ccs + i] = similarity;
            assert(similarity_table[i * num_ccs + j] >= 0);
        }
    }

    int remaining = num_ccs;

    while (remaining > 1) {
        assert (adjacency[188 * num_ccs + 410] == 0);
        int best_i = -1;
        int best_j = -1;
        float best_similarity = 0;
        for (i = 0; i < remaining; ++i) {
            for (j = i + 1; j < remaining; ++j) {
                uint8_t adjacent = adjacency[i * num_ccs + j];
                if (adjacent == 0) {
                    continue;
                }
                float similarity = similarity_table[i * num_ccs + j];
                if (similarity > best_similarity) {
                    best_similarity = similarity;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i == -1) {
            PyErr_SetString(PyExc_RuntimeError, "failed to build tree");
            return NULL;
        }

        // update regions, histograms, counts, adjacency, similarity
        regions[best_i].top = min(regions[best_i].top, regions[best_j].top);
        regions[best_i].bottom = max(regions[best_i].bottom, regions[best_j].bottom);
        regions[best_i].left = min(regions[best_i].left, regions[best_j].left);
        regions[best_i].right = max(regions[best_i].right, regions[best_j].right);
        regions[best_j] = regions[remaining - 1];

        PyObject * new_region = Py_BuildValue("iiiii", regions[best_i].left, regions[best_i].right, regions[best_i].top, regions[best_i].bottom, num_ccs-remaining+1);
        PyList_Append(region_list, new_region);
        Py_DECREF(new_region);

        for (i = 0; i < 75; ++i) {
            histogram[75*best_i + i] = (counts[best_i] * histogram[75*best_i + i] + counts[best_j] * histogram[75*best_j + i])/(counts[best_i] + counts[best_j]);
            memcpy(histogram + 75 * best_j, histogram + 75 * (remaining - 1), 75 * sizeof(float));
        }
        counts[best_i] += counts[best_j];
        counts[best_j] = counts[remaining-1];

        for (i = 0; i < remaining; ++i) {
            adjacency[best_i * num_ccs + i] = adjacency[best_i * num_ccs + i] || adjacency[best_j * num_ccs + i];
            adjacency[best_j * num_ccs + i] = adjacency[(remaining-1) * num_ccs + i];
        }
        //memcpy(adjacency + best_j * num_ccs + i, adjacency + (remaining-1) * num_ccs + i, remaining * sizeof(uint8_t));

        for (i = 0; i < remaining; ++i) {
            adjacency[i * num_ccs + best_i] = adjacency[i * num_ccs + best_i] || adjacency[i * num_ccs + best_j];
            adjacency[i * num_ccs + best_j] = adjacency[i * num_ccs + (remaining-1)];
        }

        memcpy(similarity_table + best_j * num_ccs + i, similarity_table + (remaining-1) * num_ccs + i, remaining * sizeof(float));
        for (i = 0; i < remaining; ++i) {
            similarity_table[i * num_ccs + best_j] = similarity_table[i * num_ccs + (remaining-1)];
        }

        for (i = 0; i < remaining; ++i) {
            uint8_t adjacent = adjacency[best_i * num_ccs + i];
            assert(adjacency[best_i * num_ccs + i] == adjacency[i * num_ccs + best_i]);
            if (!adjacent) {
                continue;
            }
            float color_sim = color_similarity (histogram + 75 * i, histogram + 75 * best_i);
            float size_sim = size_similarity (counts[i], counts[best_i], size);
            float fill_sim = fill_similarity (regions + i, regions + best_i, counts[i], counts[best_i], size);
            float similarity = a1 * color_sim + a2 * size_sim + a3 * fill_sim;
            similarity_table[i * num_ccs + best_i] = similarity;
            similarity_table[best_i * num_ccs + i] = similarity;
        }

        --remaining;
    }

    universe_destroy (u);

    free (similarity_table);
    free (adjacency);
    free (component_map);
    free (components);
    free (histogram);
    free (regions);
    return Py_BuildValue("NN", pyoutput, region_list);
}

static PyObject * rgb2hsv (PyObject * self, PyObject * args) {
    PyObject * pyimage;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pyimage)) {
        return NULL;
    }

    PyArrayObject * image = (PyArrayObject*)pyimage;

    if (PyArray_NDIM(image) != 3) {
        PyErr_SetString(PyExc_TypeError, "must be 3d array");
        return NULL;
    }

    npy_intp * dims = PyArray_DIMS(image);

    if (dims[2] != 3) {
        PyErr_SetString(PyExc_TypeError, "must be a 3 channel image");
        return NULL;
    }

    if (PyArray_TYPE(image) != NPY_UBYTE) {
        PyErr_SetString(PyExc_TypeError, "must be a ubyte array");
        return NULL;
    }

    PyObject * pyoutput = PyArray_SimpleNew(3, dims, NPY_FLOAT);
    PyArrayObject * output = (PyArrayObject*)pyoutput;

    int i, j;
    for (i = 0; i < dims[0]; ++i) {
        for (j = 0; j < dims[1]; ++j) {
            uint8_t r = *(uint8_t*)PyArray_GETPTR3(image, i, j, 0);
            uint8_t g = *(uint8_t*)PyArray_GETPTR3(image, i, j, 1);
            uint8_t b = *(uint8_t*)PyArray_GETPTR3(image, i, j, 2);

            float r_prime = r / 255.0;
            float g_prime = g / 255.0;
            float b_prime = b / 255.0;

            float cmax = maxf(r_prime, maxf(g_prime, b_prime));
            float cmin = minf(r_prime, minf(g_prime, b_prime));

            float v = cmax;
            if (cmin == cmax) {
                *(float*)PyArray_GETPTR3(output, i, j, 0) = 0;
                *(float*)PyArray_GETPTR3(output, i, j, 1) = 0;
                *(float*)PyArray_GETPTR3(output, i, j, 2) = v;
                continue;
            }

            float d = 0;
            float factor = 0;

            if (r_prime == cmin) {
                d = g_prime - b_prime;
                factor = 3;
            } else if (g_prime == cmin) {
                d = b_prime - r_prime;
                factor = 5;
            } else if (b_prime == cmin) {
                d = r_prime - g_prime;
                factor = 1;
            } else {
                assert(0);
            }

            float h = 60 * (factor - d/(cmax - cmin));
            float s = (cmax - cmin)/cmax;

            assert (h >= 0);
            assert (h <= 360);
            assert (s >= 0);
            assert (s <= 1);
            assert (v >= 0);
            assert (v <= 1);

            *(float*)PyArray_GETPTR3(output, i, j, 0) = h / 360.0 * 255.0;
            *(float*)PyArray_GETPTR3(output, i, j, 1) = s * 255.0;
            *(float*)PyArray_GETPTR3(output, i, j, 2) = v * 255.0;
        }
    }
    return Py_BuildValue("N", pyoutput);
}

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_regionproposals",
    "Native bits for region proposals.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION < 3
static PyMethodDef _regionproposals_methods[] = {
    {"segment", (PyCFunction)segment, METH_VARARGS|METH_KEYWORDS, "Segments an image."},
    {"rgb2hsv", (PyCFunction)rgb2hsv, METH_VARARGS, "Convert RGB image to HSV."},
    {NULL}
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit__regionproposals(void)
#else
PyMODINIT_FUNC
init_regionproposals(void)
#endif
{
    import_array();

#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
#else
    Py_InitModule3("_regionproposals", _regionproposals_methods, "Native bits for region proposal.");
#endif

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
