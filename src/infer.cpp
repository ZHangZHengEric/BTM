#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#include "str_util.h"
#include "infer.h"
#include <boost/algorithm/string.hpp>
#include "str_util.h"

using namespace str_util;

Infer::Infer(const string & type, const Pvec<double> & pz, const Pmat<double> & pw_z, int K) :
    type(type), pz(pz), pw_z(pw_z), K(K)
  {
	  
  }

Pmat<double> Infer::predict(const std::vector<std::string> & docs, const std::unordered_map<std::string, int> & w2id) {
	std::vector<Pvec<double>> predictions;
	for (const std::string& line : docs) {		
    	std::vector<int> word_idx;
    	std::vector<std::string> words = split(line);
		for (std::string w : words) {
      		w = trim(w);
		    if (w2id.find(w) == w2id.end()) {
        		word_idx.push_back(0);
      		} else {
      			word_idx.push_back(w2id.at(w));
			}
		}
		Doc doc(word_idx);
		Pvec<double> pz_d(K);
		doc_infer(doc, pz_d);
		predictions.push_back(pz_d);
	}
	return Pmat<double>(predictions);
}


void Infer::doc_infer(const Doc& doc, Pvec<double>& pz_d) {
  if (type == "sum_b")
	doc_infer_sum_b(doc, pz_d);
  else if (type == "sub_w")
	doc_infer_sum_w(doc, pz_d);
  else if (type == "mix")
	doc_infer_mix(doc, pz_d);
  else {
	cout << "[Err] unkown infer type:" << type << endl;
	exit(1);
  }
}


// p(z|d) = \sum_b{ p(z|b)p(b|d) }
void Infer::doc_infer_sum_b(const Doc& doc, Pvec<double>& pz_d) {
  pz_d.assign(K, 0);

  if (doc.size() == 1) {
	// doc is a single word, p(z|d) = p(z|w) \propo p(z)p(w|z)
	for (int k = 0; k < K; ++k)
	  pz_d[k] = pz[k] * pw_z[k][doc.get_w(0)];
  }
  else {
	// more than one words
	vector<Biterm> bs;
	doc.gen_biterms(bs);

	int W = pw_z.cols();
	for (int b = 0; b < bs.size(); ++b) {
	  int w1 = bs[b].get_wi();
	  int w2 = bs[b].get_wj();

	  // filter out-of-vocabulary words
	  if (w2 >= W) continue;

	  // compute p(z|b) \propo p(w1|z)p(w2|z)p(z)
	  Pvec<double> pz_b(K);
	  for (int k = 0; k < K; ++k) {
		assert(pw_z[k][w1]>0 && pw_z[k][w2]>0);
		pz_b[k] = pz[k] * pw_z[k][w1] * pw_z[k][w2];
	  }
	  pz_b.normalize();

	  // sum for b, p(b|d) is unifrom
	  for (int k = 0; k < K; ++k)
		pz_d[k] += pz_b[k];
	}
  }

  pz_d.normalize();
}

// p(z|d) = \sum_w{ p(z|w)p(w|d) }
void Infer::doc_infer_sum_w(const Doc& doc, Pvec<double>& pz_d) {
  pz_d.assign(K, 0);

  int W = pw_z.cols();
  const vector<int>& ws = doc.get_ws();

  for (int i = 0; i < ws.size(); ++i) {
	int w = ws[i];
	if (w >= W) continue;

	// compute p(z|w) \propo p(w|z)p(z)
	Pvec<double> pz_w(K);
	for (int k = 0; k < K; ++k)
	  pz_w[k] = pz[k] * pw_z[k][w];

	pz_w.normalize();

	// sum for b, p(b|d) is unifrom
	for (int k = 0; k < K; ++k)
	  pz_d[k] += pz_w[k];
  }
  pz_d.normalize();
}

void Infer::doc_infer_mix(const Doc& doc, Pvec<double>& pz_d) {
  pz_d.resize(K);
  for (int k = 0; k < K; ++k)
	pz_d[k] = pz[k];

  const vector<int>& ws = doc.get_ws();
  int W = pw_z.cols();
  for (int i = 0; i < ws.size(); ++i) {
	int w = ws[i];
	if (w >= W) continue;

	for (int k = 0; k < K; ++k)
	  pz_d[k] *= (pw_z[k][w] * W);
  }

	// sum for b, p(b|d) is unifrom
  pz_d.normalize();
}

// compute p(z|d, w) \proto p(w|z)p(z|d)
void Infer::compute_pz_dw(int w, const Pvec<double>& pz_d, Pvec<double>& p) {
  p.resize(K);

  for (int k = 0; k < K; ++k)
	p[k] = pw_z[k][w] * pz_d[k];

  p.normalize();
}
