// Nexuss-Fusion optional high-performance fallback kernels.
//
// Build (requires pybind11 + Eigen headers):
//   python -m pip install pybind11
//   g++ -O3 -march=native -shared -fPIC \
//       -I$(python3 -c "import pybind11; print(pybind11.get_include())") \
//       -I/usr/include/eigen3 \
//       cpp/eigen_kernels.cpp -o nexuss_fusion/backend/_eigen_native$(python3-config --extension-suffix)
//
// Imported through nexuss_fusion/backend/_eigen.py; used as a validated
// drop-in for hot non-trainable math.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>

namespace py = pybind11;

using Mat = Eigen::MatrixXd;

static Mat to_mat(py::array_t<double, py::array::c_style | py::array::forcecast> src) {
    auto buf = src.request();
    return Eigen::Map<const Mat>(static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1]);
}

static py::array_t<double> from_mat(const Mat& m) {
    py::array_t<double> out({m.rows(), m.cols()});
    Eigen::Map<Mat>(static_cast<double*>(out.request().ptr), m.rows(), m.cols()) = m;
    return out;
}

py::tuple procrustes(py::array_t<double, py::array::c_style | py::array::forcecast> source,
                     py::array_t<double, py::array::c_style | py::array::forcecast> target) {
    const Mat S = to_mat(source);
    const Mat T = to_mat(target);
    if (S.rows() != T.rows()) throw std::runtime_error("paired matrices must share the number of rows");
    Mat corr = T.transpose() * S;
    Eigen::JacobiSVD<Mat> svd(corr, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Mat R = svd.matrixV() * svd.matrixU().transpose();
    double scale = svd.singularValues().sum() / (S.array().square().sum());
    return py::make_tuple(from_mat(R), scale);
}

py::array_t<double> whiten(py::array_t<double, py::array::c_style | py::array::forcecast> h,
                           py::array_t<double, py::array::c_style | py::array::forcecast> mean,
                           py::array_t<double, py::array::c_style | py::array::forcecast> sd,
                           py::array_t<double, py::array::c_style | py::array::forcecast> scale) {
    const Mat H = to_mat(h);
    const Mat M = to_mat(mean).transpose();
    const Mat SD = to_mat(sd).transpose();
    const Mat SC = to_mat(scale).transpose();
    Mat out = (H.rowwise() - M.row(0)).array() / SD.row(0).array() * SC.row(0).array();
    return from_mat(out);
}

PYBIND11_MODULE(_eigen_native, m) {
    m.doc() = "Nexuss-Fusion Eigen fallback kernels";
    m.def("procrustes", &procrustes, "Orthogonal Procrustes (R, scale)", py::arg("source"), py::arg("target"));
    m.def("whiten", &whiten, "Diagonal whitening transform", py::arg("h"), py::arg("mean"), py::arg("std"),
          py::arg("scale"));
}