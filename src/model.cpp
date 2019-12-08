#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <limits>
#include <string>

#include "infer.h"
#include "model.h"
#include "sampler.h"
#include "str_util.h"
#include <boost/algorithm/string.hpp>
#include <boost/python.hpp>
namespace bn = boost::python::numpy;
using namespace boost;
using namespace str_util;

template <class K, class V>
boost::python::dict to_python(std::unordered_map<K, V> map) {
  typename std::unordered_map<K, V>::iterator iter;
  boost::python::dict dictionary;
  for (iter = map.begin(); iter != map.end(); ++iter) {
    dictionary[iter->first] = iter->second;
  }
  return dictionary;
}

template <typename T>
inline std::vector<T> to_std_vector(const python::list &l) {
  T elem;
  std::vector<T> elements;
  for (int i = 0; i < len(l); i++) {
    elements.push_back(python::extract<T>(l[i]));
  }
  return elements;
}
bn::ndarray to_numpy(const Pmat<double> &pw_z) {
  int M = pw_z.M;
  int N = pw_z.N;
  Py_intptr_t shape[2] = {(Py_intptr_t)M, (Py_intptr_t)N};

  bn::ndarray result = bn::zeros(2, shape, bn::dtype::get_builtin<double>());
  for (int k = 0; k < M; k++) {
    for (int w = 0; w < N; w++) {
      result[k][w] = pw_z[k][w];
    }
  }
  return result;
}

BOOST_PYTHON_MODULE(btm) {
  bn::initialize();
  python::class_<Model>("Model",
                        python::init<int, double, double, int, int, bool>())
      .def("fit", &Model::run_python)
      .def("get_pz", &Model::get_pz_py)
      .def("get_pw_z", &Model::get_pw_z_py)
      .def("predict", &Model::predict_py)
      .def("vocabulary", &Model::vocabulary_py);
}

Model::Model(int K, double a, double b, int n_iter, int save_step,
             bool has_b = false)
    : K(K), alpha(a), beta(b), n_iter(n_iter), has_background(has_b),
      save_step(save_step) {}
void Model::run_python(const python::list &documents) {
  this->run(to_std_vector<std::string>(documents));
}
void Model::run(const std::vector<string> &documents) {
  this->load_docs(documents);
  this->model_init();

  for (int it = 1; it < n_iter + 1; ++it) {
    for (int b = 0; b < bs.size(); ++b) {
      update_biterm(bs[b]);
    }
  }
}

void Model::model_init() {
  srand(time(NULL));
  // random initialize
  for (vector<Biterm>::iterator b = bs.begin(); b != bs.end(); ++b) {
    int k = Sampler::uni_sample(K);
    assign_biterm_topic(*b, k);
  }
}

void Model::build_vocabulary(const std::vector<std::string> &documents) {
  for (const std::string &line : documents) {
    std::vector<std::string> words = split(line);
    for (std::string w : words) {
      w = trim(w);
      if (this->w2id.find(w) == this->w2id.end()) {
        this->w2id[w] = this->w2id.size();
      }
    }
  }
}
Doc Model::build_doc(const std::string & line) {
  std::vector<std::string> words = split(line);
  std::vector<int> word_idx;
  for (std::string w : words) {
      w = trim(w);
      if (this->w2id.find(w) == this->w2id.end()) {
        word_idx.push_back(0);
      } else {
        word_idx.push_back(this->w2id[w]);
      }
  }
  return Doc(word_idx);
}

// input, each line is a doc
// format: wid  wid  wid ...
void Model::load_docs(const std::vector<std::string> &documents) {

  this->build_vocabulary(documents);
  this->W = w2id.size();
  pw_b.resize(W);
  nwz.resize(K, W);
  nb_z.resize(K);
  for (const std::string &line : documents) {
    Doc doc = this->build_doc(line);
    doc.gen_biterms(bs);
    for (int i = 0; i < doc.size(); ++i) {
      int w = doc.get_w(i);
      pw_b[w] += 1;
    }
  }

  pw_b.normalize();
}

// sample procedure for ith biterm
void Model::update_biterm(Biterm &bi) {
  reset_biterm_topic(bi);

  // compute p(z|b)
  Pvec<double> pz;
  compute_pz_b(bi, pz);

  // sample topic for biterm b
  int k = Sampler::mult_sample(pz.to_vector());
  assign_biterm_topic(bi, k);
}

// reset topic assignment of biterm i
void Model::reset_biterm_topic(Biterm &bi) {
  int k = bi.get_z();
  // not is the background topic
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();

  nb_z[k] -= 1;    // update number of biterms in topic K
  nwz[k][w1] -= 1; // update w1's occurrence times in topic K
  nwz[k][w2] -= 1;
  assert(nb_z[k] > -10e-7 && nwz[k][w1] > -10e-7 && nwz[k][w2] > -10e-7);
  bi.reset_z();
}

// compute p(z|w_i, w_j)
void Model::compute_pz_b(Biterm &bi, Pvec<double> &pz) {
  pz.resize(K);
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();

  double pw1k, pw2k, pk;
  for (int k = 0; k < K; ++k) {
    // avoid numerical problem by mutipling W
    if (has_background && k == 0) {
      pw1k = pw_b[w1];
      pw2k = pw_b[w2];
    } else {
      pw1k = (nwz[k][w1] + beta) / (2 * nb_z[k] + W * beta);
      pw2k = (nwz[k][w2] + beta) / (2 * nb_z[k] + 1 + W * beta);
    }
    pk = (nb_z[k] + alpha) / (bs.size() + K * alpha);
    pz[k] = pk * pw1k * pw2k;
  }

  // pz.normalize();
}

// assign topic k to biterm i
void Model::assign_biterm_topic(Biterm &bi, int k) {
  bi.set_z(k);
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();
  nb_z[k] += 1;
  nwz[k][w1] += 1;
  nwz[k][w2] += 1;
}

// p(z) is determinated by the overall proportions
// of biterms in it
Pvec<double> Model::get_pz() const {
  Pvec<double> pz(nb_z);
  pz.normalize(alpha);
  return pz;
}
bn::ndarray Model::get_pz_py() const {
  Pvec<double> pz = this->get_pz();
  Py_intptr_t shape[1] = {(Py_intptr_t)pz.size()};
  bn::ndarray result = bn::zeros(1, shape, bn::dtype::get_builtin<double>());
  std::copy(pz.p.begin(), pz.p.end(),
            reinterpret_cast<double *>(result.get_data()));
  return result;
}
Pmat<double> Model::get_pw_z() const {
  Pmat<double> pw_z(K, W); // p(w|z) = phi, size K * M
  for (int k = 0; k < K; k++) {
    for (int w = 0; w < W; w++) {
      pw_z[k][w] = (nwz[k][w] + beta) / (nb_z[k] * 2 + W * beta);
    }
  }
  return pw_z;
}
bn::ndarray Model::get_pw_z_py() const { return to_numpy(this->get_pw_z()); }
Pvec<double> Model::predict(const std::string &s,
                            const std::string &ttype = "sum_b") {
  std::vector<std::string> documents(1);
  documents[0] = s;
  return this->predict(documents, ttype)[0];
}
Pmat<double> Model::predict(const std::vector<std::string> &s,
                            const std::string &ttype = "sum_b") {
  Infer infer(ttype, this->get_pz(), this->get_pw_z(), K);
  return infer.predict(s, this->w2id);
}

bn::ndarray Model::predict_py(const python::list &pydocuments,
                              const std::string &ttype = "sum_b") {

  Infer infer(ttype, this->get_pz(), this->get_pw_z(), K);
  auto documents = to_std_vector<std::string>(pydocuments);
  return to_numpy(infer.predict(documents, this->w2id));
}
boost::python::dict Model::vocabulary_py() const {
  return to_python(this->w2id);
}