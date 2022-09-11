#include "include/cusr.h"
#include <fstream>

using namespace std;
using namespace cusr;

vector<vector<float>> dataset;
vector<float> label;

void gen_pagie_uniform(int dim);
void split(const std::string &s, std::vector<std::string> &v, const std::string &c);
void load_csv_dataset(const string &file_name, vector<vector<float>> &dataset, vector<float> &label, int total);
void run_airline();
void run_year();

#define LOG_FILE_1 "CUSR_time.csv"
#define LOG_FILE_2 "CUSR_fit.csv"
#define YEAR "YearPredictionMSD.csv"
#define AIRLINE "airline_reg.csv"

void pagie_30_runs_avg();


int main() {
    pagie_30_runs_avg();
    run_year();
    run_airline();
    return 0;
}


void run_year()
{
    set_constant_prob(0.05);
    load_csv_dataset(YEAR, dataset, label, 515072);
    float time_total = 0;
    float fit_total = 0;
    for (int i = 0; i < 3; i++) {
        RegressionEngine reg;
        reg.function_set = {
                _add, _cos, _sub, _div, _tan, _mul, _sin
        };
        reg.use_gpu = true;
        reg.max_program_depth = 10;
        reg.population_size = 50;
        reg.generations = 50;
        reg.parsimony_coefficient = 0;
        reg.const_range = {-5, 5};
        reg.init_depth = {4, 10};
        reg.parsimony_coefficient = 0;
        reg.tournament_size = 3;
        reg.metric = metric_t::root_mean_square_error;
        reg.fit(dataset, label);
        time_total += reg.regress_time_in_sec;
        fit_total += reg.best_program.fitness;
    }
    ofstream outfile(LOG_FILE_1, ios::app);
    outfile << "year" << "," << time_total / 3 << "," << fit_total / 3 << endl;
}


void run_airline()
{
    set_constant_prob(0.09);
    load_csv_dataset(AIRLINE, dataset, label, 5810176);
    float time_total = 0;
    float fit_total = 0;
    for (int i = 0; i < 3; i++) {
        RegressionEngine reg;
        reg.function_set = {
                _add, _cos, _sub, _div, _tan, _mul, _sin
        };
        reg.use_gpu = true;
        reg.max_program_depth = 10;
        reg.population_size = 50;
        reg.generations = 50;
        reg.parsimony_coefficient = 0;
        reg.const_range = {-5, 5};
        reg.init_depth = {4, 10};
        reg.parsimony_coefficient = 0;
        reg.tournament_size = 3;
        reg.metric = metric_t::root_mean_square_error;
        reg.fit(dataset, label);
        time_total += reg.regress_time_in_sec;
        fit_total += reg.best_program.fitness;
    }
    ofstream outfile(LOG_FILE_1, ios::app);
    outfile << "airline" << "," << time_total / 3 << "," << fit_total / 3 << endl;
}


void pagie_fitness(int dim) {
    gen_pagie_uniform(dim);
    float fit_each_gen[50] = {0};
    for (int i = 0; i < 10; i++) {
        RegressionEngine reg;
        reg.function_set = {
                _add, _cos, _sub, _div, _tan, _mul, _sin
        };
        reg.use_gpu = true;
        reg.max_program_depth = 10;
        reg.population_size = 100;
        reg.generations = 50;
        reg.parsimony_coefficient = 0;
        reg.const_range = {-5, 5};
        reg.init_depth = {4, 10};
        reg.tournament_size = 3;
        reg.metric = metric_t::root_mean_square_error;
        reg.fit(dataset, label);
        for (int j = 0; j < 50; j++) {
            fit_each_gen[j] += reg.best_program_in_each_gen[j].fitness;
        }
    }
    ofstream outfile(LOG_FILE_2, ios::app);
    outfile << dim;
    for (int i = 0; i < 50; i++) {
        float avg = fit_each_gen[i] / 10.0f;
        outfile << "," << avg;
    }
    outfile << endl;
}


void gen_pagie_uniform(int dim) {
    dataset.clear();
    label.clear();
    float xIn = 10.0 / dim;
    float yIn = 10.0 / dim;
    for (float x = -5; x < 5; x += xIn) {
        for (float y = -5; y < 5; y += yIn) {
            vector<float> temp;
            temp.emplace_back(x);
            temp.emplace_back(y);
            dataset.emplace_back(temp);
            label.emplace_back(x * x * x * x / (x * x * x * x + 1) + y * y * y * y / (y * y * y * y + 1));
        }
    }
}


void split(const std::string &s, std::vector<std::string> &v, const std::string &c) {
    v.clear();
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}


void load_csv_dataset(const string &file_name, vector<vector<float>> &dataset, vector<float> &label, int total) {
    dataset.clear();
    label.clear();

    cout << "> loading dataset" << endl;

    ifstream in(file_name, ios::in);
    string line;

    // 检查数据的行数
    getline(in, line);
    vector<string> vec_line;
    split(line, vec_line, ",");
    int rows = vec_line.size();
    in.seekg(0, ios::beg);
    int line_count = 0;
    // 读每行
    while (getline(in, line)) {
        split(line, vec_line, ",");
        vector<float> f_line;
        for (int i = 0; i < rows - 1; ++i) {
            f_line.emplace_back(std::stof(vec_line[i]));
        }
        dataset.emplace_back(f_line);
        label.emplace_back(std::stof(vec_line[rows - 1]));

        if (++line_count == total) {
            break;
        }
    }
    cout << "> complete" << endl;
}


void pagie_30_runs_avg() {
    int dim = 64;
    for (int i = 0; i < 7; ++i) {
        gen_pagie_uniform(dim);

        ofstream outfile_time(LOG_FILE_1, ios::app);
        ofstream outfile_fit (LOG_FILE_2, ios::app);

        for (int run = 0; run < 30; run++) {
            RegressionEngine reg;
            reg.function_set = {
                    _add, _cos, _sub, _div, _tan, _mul, _sin
            };
            reg.use_gpu = true;
            reg.max_program_depth = 10;
            reg.population_size = 50;
            reg.generations = 50;
            reg.parsimony_coefficient = 0;
            reg.const_range = {-5, 5};
            reg.init_depth = {4, 10};
            reg.tournament_size = 3;
            reg.metric = metric_t::root_mean_square_error;
            reg.fit(dataset, label);
            outfile_time << reg.regress_time_in_sec << ",";
            outfile_fit << reg.best_program.fitness << ",";
        }
        outfile_time << endl;
        outfile_fit << endl;
        dim *= 2;
    }
}




