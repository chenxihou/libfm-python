#ifndef FTRL_LIBRARY_H
#define FTRL_LIBRARY_H


extern "C" {

int ftrl_classification = 0;
int ftrl_regression = 1;

struct csr_binary_matrix {
    int *columns;
    int *indptr;
    float *data;
    int num_examples;
};

struct ftrl_params {
    float alpha;
    float beta;
    float l1;
    float l2;
    int model_type;
};

struct ftrl_model {
    float n_intercept;
    float z_intercept;
    float w_intercept;

    float *n;
    float *z;
    float *w;
    
    float *nf;
    float *zf;
    float *wf;
    
    float *sum_f;

    int num_factors;
    int num_features;
    ftrl_params params;
    
    int d0;
    int d1;
    int d2;
};



ftrl_model ftrl_init_model(ftrl_params &params, int num_features, int num_factors, int d0, int d1, int d2);

void ftrl_model_cleanup(ftrl_model *model);

float ftrl_fit(int *values, float *data, int len, float y, ftrl_model *model);

float ftrl_fit_batch(csr_binary_matrix &X, float *target, int num_examples,
                     ftrl_model *model, bool shuffle);

float ftrl_predict(int *values, float *data, int len, ftrl_model *model);

void ftrl_predict_batch(csr_binary_matrix &X, ftrl_model *model,
                        float *result);

void ftrl_weights(ftrl_model *model, float *weights, float *intercept);

void ftrl_save_model(char *path, ftrl_model *model);

ftrl_model ftrl_load_model(char *path);

};

#endif
