#ifndef _INFERLDA_H
#define _INFERLDA_H

#include <string>

#include "pvec.h"
#include "pmat.h"

#include "doc.h"
#include <unordered_map>
using namespace std;

class Infer {
private:
 
  string type;			// infer type

  Pvec<double> pz;	    // p(z) = theta
  Pmat<double> pw_z;   // p(w|z) = phi, size K * M
  int K;
private:


  void doc_infer(const Doc& doc, Pvec<double>& pz_d);
  void doc_infer_sum_b(const Doc& doc, Pvec<double>& pz_d);
  void doc_infer_sum_w(const Doc& doc, Pvec<double>& pz_d);
  void doc_infer_mix(const Doc& doc, Pvec<double>& pz_d);

  // compute condition distribution p(z|w, d) with p(w|z) fixed
  void compute_pz_dw(int w, const Pvec<double>& pz_d, Pvec<double>& p);

public:
  Infer(const string & type, const Pvec<double> & pz, const Pmat<double> & pw_z, int K);

  Pmat<double> predict(const std::vector<std::string> & docs, const std::unordered_map<std::string, int> & w2id);
};

#endif
