#include <cstdio>
#include <vector>

using namespace std;

inline float step_function(float x) {
    if (x > 0)
        return 1.0;
    return 0.0;
}
inline float weighted_sum(const vector<float> w, const vector<float>& x, float bias) {
    float sum = bias;
    for (int i = 0; i < w.size(); i++) {
        sum += w[i] * x[i];
    }
    return sum;
}
inline void train(vector<float>& w, float& bias, vector<vector<float> >& input_vecs, vector<float>& labels, int iteration, float rate) {
    for (int i = 0; i < w.size(); i++)
        w[i] = 0.0;
    bias = 0.0;

    for (int i = 0; i < iteration; i++) {
        for (int j = 0; j < input_vecs.size(); j++) {
            float predict_output = step_function(weighted_sum(w, input_vecs[j], bias));
            float delta = labels[j] - predict_output;

            for (int k = 0; k < w.size(); k++) {
                w[k] += rate * delta * input_vecs[j][k];
            }
            bias += rate * delta;
        }
    }
}
inline void print_w(vector<float>& w) {
    printf("\n");
    printf("weights = [");
    for (int i = 0; i < w.size(); i++) {
        printf("%f, ", w[i]);
    }
    printf("]");
    printf("\n");
}
int main() {
    vector<float> w(2, 0.0);
    float bias;
    vector<vector<float> > input_vecs;
    vector<float> labels;

    vector<float> tmp(2, 0.0);
    
    tmp[0] = 1.0; tmp[1] = 1.0;
    input_vecs.push_back(tmp);
    labels.push_back(1.0);
    tmp[0] = 0.0; tmp[1] = 0.0;
    input_vecs.push_back(tmp);
    labels.push_back(0.0);
    tmp[0] = 1.0; tmp[1] = 0.0;
    input_vecs.push_back(tmp);
    labels.push_back(0.0);
    tmp[0] = 0.0; tmp[1] = 1.0;
    input_vecs.push_back(tmp);
    labels.push_back(0.0);




    train(w, bias, input_vecs, labels, 100, 0.1);

    for (int i = 0; i < input_vecs.size(); i++) {
        printf("%f and %f = %f\n", input_vecs[i][0], input_vecs[i][1], step_function(weighted_sum(w, input_vecs[i], bias)));
    }

    print_w(w);
    printf("bias = %f\n", bias);
    return 0;
}