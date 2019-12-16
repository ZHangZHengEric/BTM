/**
 * Biterm topic model(BTM) with Gbbis sampling
 * Author: Xiaohui Yan(xhcloud@gmail.com)
 * 2012-9-25
 */
#ifndef _MODEL_H
#define _MODEL_H

#include "biterm.h"
#include "boost/python/numpy.hpp"
#include "doc.h"
#include "pmat.h"
#include "pvec.h"
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <fstream>
#include <vector>
#include <unordered_map>

using namespace std;

class Model {
public:
  vector<Biterm> bs;

protected:
  int W;      // vocabulary size
  int K;      // number of topics
  int n_iter; // maximum number of iteration of Gibbs Sampling
  int save_step;

  double alpha; // hyperparameters of p(z)
  double beta;  // hyperparameters of p(w|z)

  // sample recorders
  Pvec<int> nb_z; // n(b|z), size K*1
  Pmat<int> nwz;  // n(w,z), size K*W

  Pvec<double> pw_b; // the background word distribution

  // If true, the topic 0 is set to a background topic that
  // equals to the emperiacal word dsitribution. It can filter
  // out common words
  bool has_background;

  // Vocabulary word to index
  std::unordered_map<std::string, int> w2id;

  bool initialized;
public:
  Model(int K, double a, double b, int n_iter, int save_step,
        bool has_b);

  // run estimate procedures
  void run_python(const boost::python::list &documents);
  void run(const std::vector<string> &documents);
  boost::python::numpy::ndarray get_pz_py() const;
  boost::python::numpy::ndarray get_pw_z_py() const;

  Pvec<double> predict(const std::string & s, const std::string & ttype);
  Pmat<double> predict(const std::vector<std::string> & s, const std::string & ttype);
  boost::python::numpy::ndarray predict_py(const boost::python::list &documents,  const std::string & ttype);
  boost::python::dict vocabulary_py() const;

  void fit_step();
private:
  // intialize memeber varibles and biterms
  void model_init(); // load from docs
  void load_docs(const std::vector<std::string> &documents);

  // update estimate of a biterm
  void update_biterm(Biterm &bi);

  // reset topic proportions for biterm b
  void reset_biterm_topic(Biterm &bi);

  // assign topic proportions for biterm b
  void assign_biterm_topic(Biterm &bi, int k);

  // compute condition distribution p(z|b)
  void compute_pz_b(Biterm &bi, Pvec<double> &p);

  Pvec<double> get_pz() const;
  Pmat<double> get_pw_z() const;
  void build_vocabulary(const std::vector<std::string> &documents);
  Doc build_doc(const std::string & line);

};

#endif
