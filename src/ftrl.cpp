#include "ftrl.h"

#include <string>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <time.h>

#if defined USEOMP
#include <omp.h>
#endif

#define TOLERANCE 1e-6f

using namespace std;

float sign(float x) {
    if (x < 0) {
        return -1.0f;
    } else {
        return 1.0f;
    }
}

float calculate_sigma(float n, float grad, float alpha) {
    return (sqrt(n + grad * grad) - sqrt(n)) / alpha;
}

float calculate_w(float z, float n, ftrl_params &params) {
    float s = sign(z);
    if (s * z <= params.l1) {
        return 0.0f;
    }

    float w = (s * params.l1 - z) / ((params.beta + sqrt(n)) / params.alpha + params.l2);
    return w;
}

float log_loss(float y, float p) {
    if (y == 1.0f) {
        return -log(fmaxf(p, TOLERANCE));
    } else if (y == -1.0f) {
        return -log(fmaxf(1 - p, 1 - TOLERANCE));
    }
}

float sigmoid(float x) {
    if (x <= -35.0f) {
        return 0.000000000000001f;
    } else if (x >= 35.0f) {
        return 0.999999999999999f;
    }

    return 1.0f / (1.0f + exp(-x));
}

float ftrl_predict(int *values, float *data, int len, ftrl_model *model) {
    float result = 0;
    int d0 = model->d0;
    int d1 = model->d1;
    int d2 = model->d2;
    ftrl_params params = model->params;
    
    
    
    if(d0 > 0){
        model->w_intercept = calculate_w(model->z_intercept, model->n_intercept, params);
        result = result + model->w_intercept;
//         printf("result[0]::%f\t", result);
    }
    
    

    if(d1 > 0){
        float *n = model->n;
        float *z = model->z;
        float *w = model->w;


        for (int k = 0; k < len; k++) {
            int i = values[k];
            float x = data[k];
//             float x = 1.0;
            w[i] = calculate_w(z[i], n[i], params);
            result = result + w[i]*x;
        }
//         printf("result1::%f\t", result);
    }
    
    
    if(d2 > 0){
        float *nf = model->nf;
        float *zf = model->zf;
        float *wf = model->wf;

        float *sum_f = model->sum_f;
        float sum, sum_sqr, d;
        int num_factors = model->num_factors;
        for(int f = 0; f < num_factors; f++) {
            sum = sum_sqr = 0.0;
            for(int k = 0; k < len; k++) {
                int i = values[k];
                float x = data[k];
//                 float x = 1.0;
                int index = i * num_factors + f;
                wf[index] = calculate_w(zf[index], nf[index], params);
                d = wf[index] * x;
                sum += d;
                sum_sqr += d * d;
                }
            sum_f[f] = sum;
            result += 0.5 * (sum * sum - sum_sqr);
            }
//         printf("result2::%f\t", result);
//         printf("sum[0]:;%f, sum[1]::%f\n", sum_f[0], sum_f[1]);
    }
    
    return result;
}

float ftrl_fit(int *values, float *data, int len, float y, ftrl_model *model) {
    int num_features = model->num_features;
    int num_factors = model->num_factors;
    int d0 = model->d0;
    int d1 = model->d1;
    int d2 = model->d2;
    ftrl_params params = model->params;
    float wtx = ftrl_predict(values, data, len, model);

//     float pred = wtx;
    float mult;
    if (params.model_type == ftrl_classification) {
//         pred = sigmoid(wtx);
        mult = y * (1 / (1 + exp(-wtx * y)) -1 ); 
    }
    else{
        mult = wtx - y; 
    }

//     float grad = pred - y;

//     float mult = y * (1 / (1 + exp(-wtx * y)) -1 );    
    
//     float grad = mult;
    float grad;
    
    if(d0 > 0){
        grad = mult;
        float sigma_intercept = calculate_sigma(model->n_intercept, grad, params.alpha);
        model->z_intercept = model->z_intercept + grad - sigma_intercept * model->w_intercept;
        model->n_intercept = model->n_intercept + grad * grad;
    }

    if(d1 > 0){
        float *n = model->n;
        float *z = model->z;
        float *w = model->w;

        for (int k = 0; k < len; k++) {
            int i = values[k];
            float x = data[k];
            grad = mult * x;
            float sigma = calculate_sigma(n[i], grad, params.alpha);
            z[i] = z[i] + grad - sigma * w[i];
            n[i] = n[i] + grad * grad;
        }
    }

    if(d2 > 0){
        float *nf = model->nf;
        float *zf = model->zf;
        float *wf = model->wf;
        float *sum_f = model->sum_f;
        int num_factors = model->num_factors;
        for (int k = 0; k < len; k++) {
            int i = values[k];
            int x = data[k];
            for(int f = 0; f < num_factors; f++) {
                int index = i * num_factors + f;
                float grad_vf = mult * (sum_f[f] * x - wf[index] * x * x);
                float sigma = calculate_sigma(nf[index], grad_vf, params.alpha);
                zf[index] = zf[index] + grad_vf - sigma * wf[index];
                nf[index] = nf[index] + grad_vf * grad_vf;
            }
        }
    }
//     printf("y::%f, pred::%f, loss::%f\n", y, pred, log_loss(y, pred));
    float loss;
    if (params.model_type == ftrl_classification) {
        loss = log_loss(y, sigmoid(wtx));
    }
    else{
        loss = 0.5 * (wtx - y) * (wtx - y);
    }
//     return log_loss(y, sigmoid(wtx));
    return loss;
}

float ftrl_fit_batch(csr_binary_matrix &X, float *target, int num_examples, ftrl_model *model, bool shuffle) {
    ftrl_params params = model->params;
    int *values = X.columns;
    int *indptr = X.indptr;
    float *data = X.data;
    
    int *idx = new int[num_examples];
    for (int i = 0; i < num_examples; i++) {
        idx[i] = i;
    }
    
    if (shuffle) {
        random_shuffle(&idx[0], &idx[num_examples]);
    }

    float loss_total = 0.0f;
    
    // todo(fit parallelize)
//     #if defined USEOMP
//     #pragma omp parallel for schedule(static) reduction(+: loss_total)
//     #endif
    
    for (int id = 0; id < num_examples; id++) {
        int i = idx[id];

        float y = target[i];
        if (params.model_type == ftrl_classification) {
            if(y > 0){
                y = 1.0;
            }
            else{
                y = -1.0;
            }
        }
        
        int *x = &values[indptr[i]];
        float *x_data = &data[indptr[i]];
        int len_x = indptr[i + 1] - indptr[i];
        float loss = ftrl_fit(x, x_data, len_x, y, model);
        loss_total = loss_total + loss;
    }

    delete[] idx;

    return loss_total / num_examples;
}

void ftrl_weights(ftrl_model *model, float *weights, float *intercept) {
    ftrl_params params = model->params;
    *intercept = calculate_w(model->z_intercept, model->n_intercept, params);

    float *z = model->z;
    float *n = model->n;

    int d = model->num_features;

    for (int i = 0; i < d; i++) {
        weights[i] = calculate_w(z[i], n[i], params);
    }
}

void ftrl_predict_batch(csr_binary_matrix &X, ftrl_model *model, float *result) {
    int n = X.num_examples;
    int *values = X.columns;
    int *indptr = X.indptr;
    float *data = X.data;
    
    
    
    ftrl_params params = model->params;
    
    // #if defined USEOMP
    // #pragma omp parallel for schedule(static)
    // #endif
    
    for (int i = 0; i < n; i++) {
        int len_x = indptr[i + 1] - indptr[i];
        int *x = &values[indptr[i]];
        float *x_data = &data[indptr[i]];
        float wtx = ftrl_predict(x, x_data, len_x, model);
        if (params.model_type == ftrl_classification) {
            result[i] = sigmoid(wtx);
//             printf("wtx::%f, result::%f\n", wtx, result[i]);
            }
        else{
            result[i] = wtx;
        }
    }
}

float *zero_float_vector(int size, bool zero) {
    float *result = new float[size];
    if(zero){
        memset(result, 0.0f, size * sizeof(float));
    }
    else{
        for(int i=0; i<size; i++){
            result[i] = rand()%100/(double)101;
        }
    }
    return result;
}

ftrl_model ftrl_init_model(ftrl_params &params, int num_features, int num_factors, int d0, int d1, int d2) {
    ftrl_model model;

    model.n_intercept = 0.0f;
    model.z_intercept = 0.0f;
    model.w_intercept = 0.0f;

    model.num_features = num_features;
    model.num_factors = num_factors;
    model.n = zero_float_vector(num_features, true);
    model.z = zero_float_vector(num_features, true);
    model.w = zero_float_vector(num_features, true);
    
    model.nf = zero_float_vector(num_features*num_factors, false);
    model.zf = zero_float_vector(num_features*num_factors, false);
    model.wf = zero_float_vector(num_features*num_factors, false);
    
    model.sum_f = zero_float_vector(num_factors, true);
    
    model.params = params;
    
    model.d0 = d0;
    model.d1 = d1;
    model.d2 = d2;
    
    return model;
}

void ftrl_model_cleanup(ftrl_model *model) {
    delete[] model->n;
    delete[] model->z;
    delete[] model->w;
    delete[] model->nf;
    delete[] model->zf;
    delete[] model->wf;
    delete[] model->sum_f;
    
}

void ftrl_save_model(char *path, ftrl_model *model) {
    FILE *f = fopen(path, "wb");

    int n = model->num_features;
    fwrite(&n, sizeof(int), 1, f);
    
    int num_factors = model->num_factors;
    fwrite(&num_factors, sizeof(int), 1, f);
    
    fwrite(model->sum_f, sizeof(float), num_factors, f);

    fwrite(model->nf, sizeof(float), n*num_factors, f);
    fwrite(model->zf, sizeof(float), n*num_factors, f);
    fwrite(model->wf, sizeof(float), n*num_factors, f);
    
    fwrite(model->n, sizeof(float), n, f);
    fwrite(model->z, sizeof(float), n, f);
    fwrite(model->w, sizeof(float), n, f);

    fwrite(&model->n_intercept, sizeof(int), 1, f);
    fwrite(&model->z_intercept, sizeof(int), 1, f);
    fwrite(&model->w_intercept, sizeof(int), 1, f);

    fwrite(&model->params, sizeof(ftrl_params), 1, f);
    
    int d0 = model->d0;
    fwrite(&d0, sizeof(int), 1, f);
    
    int d1 = model->d1;
    fwrite(&d1, sizeof(int), 1, f);
    
    int d2 = model->d2;
    fwrite(&d2, sizeof(int), 1, f);

    fclose(f);
}

ftrl_model ftrl_load_model(char *path) {
    ftrl_model model;

    FILE *f = fopen(path, "rb");
    int n = 0;
    fread(&n, sizeof(int), 1, f);
    model.num_features = n;
    
    int num_factors = 0;
    fread(&num_factors, sizeof(int), 1, f);
    model.num_factors = num_factors;
    
    model.sum_f = new float[num_factors];
    fread(model.sum_f, sizeof(float), num_factors, f);
    
    model.nf = new float[n*num_factors];
    model.zf = new float[n*num_factors];
    model.wf = new float[n*num_factors];
    
    fread(model.nf, sizeof(float), n*num_factors, f);
    fread(model.zf, sizeof(float), n*num_factors, f);
    fread(model.wf, sizeof(float), n*num_factors, f);

    model.n = new float[n];
    model.z = new float[n];
    model.w = new float[n];

    fread(model.n, sizeof(float), n, f);
    fread(model.z, sizeof(float), n, f);
    fread(model.w, sizeof(float), n, f);

    fread(&model.n_intercept, sizeof(int), 1, f);
    fread(&model.z_intercept, sizeof(int), 1, f);
    fread(&model.w_intercept, sizeof(int), 1, f);

    fread(&model.params, sizeof(ftrl_params), 1, f);

    int d0 = 0;
    fread(&d0, sizeof(int), 1, f);
    model.d0 = d0;
    
    int d1 = 0;
    fread(&d1, sizeof(int), 1, f);
    model.d1 = d1;
    
    int d2 = 0;
    fread(&d2, sizeof(int), 1, f);
    model.d2 = d2;
    
    
    fclose(f);

    return model;
}
