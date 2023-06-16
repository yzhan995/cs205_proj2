#include <vector>
using namespace std;

struct classifier {
int instances, features; // number of instances, number of features
    vector<double> label; //label of each instance
    vector<vector<double>> feature; //features of each instance

    classifier(): instances(0), features(0) { }

    void parse_file(char *filename);
    void forward_selection();
    void backward_elimination();
    double cross_validation(int *current_features); // Calculate the accuracy of the current feature set
};