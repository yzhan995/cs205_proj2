#include "feature_selection.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <cmath>
char filename[100];

void classifier::parse_file(char *filename) {
    ifstream infile;
    infile.open(filename);
    string line;
    while(getline(infile, line)) {
        vector<double> tmpv;
        double tmp;
        istringstream cline(line);
        cline >> tmp;
        label.push_back(tmp);
        while (cline >> tmp)
            tmpv.push_back(tmp);
        feature.push_back(tmpv);
    }
    features = feature[0].size();
    instances = label.size();
    printf("Finish parse file:\n\tNumber of instances: %d\n\tNumber of features: %d\n", instances, features);
}

double classifier::cross_validation(int *current_features) { // Calculate the accuracy of the current feature set
    int corrects = 0;
    for (int i = 0; i < instances; i++) {
        double min_dist = 1e18;
        int nearest = -1;
        for (int j = 0; j < instances; j++) { // Find the nearest node
            if (i == j) continue;
            double dist = 0;
            for (int k = 0; k < features; k++)
                if (current_features[k]) dist += pow(feature[i][k] - feature[j][k], 2);
            dist = sqrt(dist);
            if (dist < min_dist) {
                min_dist = dist, nearest = j;
            }
        }
        if (label[i] == label[nearest]) ++corrects; // Compare the nearest pair
    }
    return 1.0 * corrects / instances;
}

void classifier::forward_selection() {
    printf("Start forward selection\n");
    double best_acc = 0;
    int* current_features = new int[features];
    int* best_features = new int[features];
    for (int i = 0; i < features; i++) // Initialize current_features as an empty set
        current_features[i] = 0;
    for (int i = 0; i < features; i++) {
        double max_acc = 0;
        int select_feature = -1;
        for (int j = 0; j < features; j++) {
            if (current_features[j]) continue;
            current_features[j] = 1; // try add j to current_features and check
            double acc = cross_validation(current_features);
            printf("\tTry add feature %d, accuracy is %.2lf\n", j, acc);
            current_features[j] = 0;
            if (acc > max_acc) {
                max_acc = acc, select_feature = j;
            }
        }
        current_features[select_feature] = 1; // add the optimal feature to current_features
        if (max_acc > best_acc) {
            best_acc = max_acc;
            for (int j = 0; j < features; j++)
                best_features[j] = current_features[j];
        }
        printf("Best feature to add: %d, now feature set is {", select_feature);
        int sign = 0;
        for (int j = 0; j < features; j++)
            if (current_features[j]) 
                if (sign == 0) printf("%d", j), sign = 1;
                else printf(", %d", j);
        printf("}\n");
    }
    printf("Finish foward selection, the best accuracy is %.2lf, best feature set is {", best_acc);
    int sign = 0;
    for (int j = 0; j < features; j++)
        if (best_features[j])
                if (sign == 0) printf("%d", j), sign = 1;
                else printf(", %d", j);
    printf("}\n");
}

void classifier::backward_elimination() {
    printf("Start backward_elimination\n");
    double best_acc = 0;
    int* current_features = new int[features];
    int* best_features = new int[features];
    for (int i = 0; i < features; i++) // Initialize current_features as a complete set
        current_features[i] = 1;
    for (int i = 0; i < features; i++) {
        double max_acc = 0;
        int select_feature = -1;
        for (int j = 0; j < features; j++) {
            if (!current_features[j]) continue;
            current_features[j] = 0; // try eliminate j from current_features and check
            double acc = cross_validation(current_features);
            printf("\tTry elminate feature %d, accuracy is %.2lf\n", j, acc);
            current_features[j] = 1;
            if (acc > max_acc) {
                max_acc = acc, select_feature = j;
            }
        }
        current_features[select_feature] = 0; // eliminate the worst feature from current_features
        if (max_acc > best_acc) {
            best_acc = max_acc;
            for (int j = 0; j < features; j++)
                best_features[j] = current_features[j];
        }
        printf("Best feature to elminate: %d, now feature set is {", select_feature);
        int sign = 0;
        for (int j = 0; j < features; j++)
            if (current_features[j]) 
                if (sign == 0) printf("%d", j), sign = 1;
                else printf(", %d", j);
        printf("}\n");
    }
    printf("Finish backward_elimination, the best accuracy is %.2lf, best feature set is {", best_acc);
    int sign = 0;
    for (int j = 0; j < features; j++)
        if (best_features[j])
                if (sign == 0) printf("%d", j), sign = 1;
                else printf(", %d", j);
    printf("}\n");
}

int main() {   
    classifier *S = new classifier();
    printf("Please enter the name of the file to test: \n");
    scanf("%s", filename);
    printf("Enter \'0\' to use Forward Selection, \'1\' to use Backward Elimination: \n");
    int tmp;
    scanf("%d", &tmp);
    auto start = std::chrono::high_resolution_clock::now();
    S->parse_file(filename);
    if (tmp == 0) S->forward_selection();
    else S->backward_elimination();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); // 计算时间差
    std::cout << "Time taken by program: " << duration.count() << " milliseconds" << std::endl; // 输出执行时间
}