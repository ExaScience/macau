#include "MatrixIO.h"

#include <array>
#include <iostream>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <array>

#include <highfive/H5Easy.hpp>

#include <Utils/Error.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

#include <SmurffCpp/IO/GenericIO.h>

namespace smurff {

#define EXTENSION_SDM ".sdm" //sparse double matrix (binary file)
#define EXTENSION_SBM ".sbm" //sparse binary matrix (binary file)
#define EXTENSION_MTX ".mtx" //sparse matrix (txt file)
#define EXTENSION_MM  ".mm"  //sparse matrix (txt file)
#define EXTENSION_CSV ".csv" //dense matrix (txt file)
#define EXTENSION_DDM ".ddm" //dense double matrix (binary file)

#define MM_OBJ_MATRIX   "MATRIX"
#define MM_FMT_ARRAY    "ARRAY"
#define MM_FMT_COORD    "COORDINATE"
#define MM_FLD_REAL     "REAL"
#define MM_FLD_PATTERN  "PATTERN"
#define MM_SYM_GENERAL  "GENERAL"

matrix_io::MatrixType matrix_io::ExtensionToMatrixType(const std::string& fname)
{
   std::size_t dotIndex = fname.find_last_of(".");
   if (dotIndex == std::string::npos)
   {
      THROWERROR("Extension is not specified in " + fname);
   }

   std::string extension = fname.substr(dotIndex);

   if (extension == EXTENSION_SDM)
   {
      return matrix_io::MatrixType::sdm;
   }
   else if (extension == EXTENSION_SBM)
   {
      return matrix_io::MatrixType::sbm;
   }
   else if (extension == EXTENSION_MTX || extension == EXTENSION_MM)
   {
       return matrix_io::MatrixType::mtx;
   }
   else if (extension == EXTENSION_CSV)
   {
      return matrix_io::MatrixType::csv;
   }
   else if (extension == EXTENSION_DDM)
   {
      return matrix_io::MatrixType::ddm;
   }
   else
   {
      THROWERROR("Unknown file type: " + extension + " specified in " + fname);
   }
   return matrix_io::MatrixType::none;
}

std::string MatrixTypeToExtension(matrix_io::MatrixType matrixType)
{
   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      return EXTENSION_SDM;
   case matrix_io::MatrixType::sbm:
      return EXTENSION_SBM;
   case matrix_io::MatrixType::mtx:
      return EXTENSION_MTX;
   case matrix_io::MatrixType::csv:
      return EXTENSION_CSV;
   case matrix_io::MatrixType::ddm:
      return EXTENSION_DDM;
   case matrix_io::MatrixType::none:
      {
         THROWERROR("Unknown matrix type");
      }
   default:
      {
         THROWERROR("Unknown matrix type");
      }
   }
   return std::string();
}

std::shared_ptr<MatrixConfig> matrix_io::read_matrix(const std::string& filename, bool isScarce)
{
   std::shared_ptr<MatrixConfig> ret;

   MatrixType matrixType = ExtensionToMatrixType(filename);

   THROWERROR_FILE_NOT_EXIST(filename);

   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = matrix_io::read_sparse_float64_bin(fileStream, isScarce);
         break;
      }
   case matrix_io::MatrixType::sbm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = matrix_io::read_sparse_binary_bin(fileStream, isScarce);
         break;
      }
   case matrix_io::MatrixType::mtx:
      {
         std::ifstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = matrix_io::read_matrix_market(fileStream, isScarce);
         break;
      }
   case matrix_io::MatrixType::csv:
      {
         std::ifstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = matrix_io::read_dense_float64_csv(fileStream);
         break;
      }
   case matrix_io::MatrixType::ddm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = matrix_io::read_dense_float64_bin(fileStream);
         break;
      }
   case matrix_io::MatrixType::none:
      {
         THROWERROR("Unknown matrix type specified in " + filename);
      }
   default:
      {
         THROWERROR("Unknown matrix type specified in " + filename);
      }
   }

   ret->setFilename(filename);

   return ret;
}

std::shared_ptr<MatrixConfig> matrix_io::read_dense_float64_bin(std::istream& in)
{
   std::uint64_t nrow;
   std::uint64_t ncol;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));
   std::vector<double> values(nrow * ncol);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   return std::make_shared<MatrixConfig>(nrow, ncol, values, NoiseConfig());
}

std::shared_ptr<MatrixConfig> matrix_io::read_dense_float64_csv(std::istream& in)
{
   // rows and cols
   std::uint64_t nrow, ncol;
   generic_io::read_line_single(in, nrow);
   generic_io::read_line_single(in, ncol);

   // file contains values row-by-row (row-major)
   std::vector<std::vector<double>> values_per_row(nrow);
   for(std::uint64_t row = 0; row<nrow; row++)
      generic_io::read_line_delim(in, values_per_row[row], ',', ncol);

   // MatrixConfig needs col-major
   std::vector<double> values_per_col;
   for(std::uint64_t col = 0; col<ncol; col++)
      for(std::uint64_t row = 0; row<nrow; row++)
         values_per_col.push_back(values_per_row[row][col]);

   return std::make_shared<MatrixConfig>(nrow, ncol, values_per_col, NoiseConfig());
}

std::shared_ptr<MatrixConfig> matrix_io::read_sparse_float64_bin(std::istream& in, bool isScarce)
{
   std::uint64_t nrow;
   std::uint64_t ncol;
   std::uint64_t nnz;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   std::vector<std::uint32_t> rows(nnz);
   in.read(reinterpret_cast<char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row--; });

   std::vector<std::uint32_t> cols(nnz);
   in.read(reinterpret_cast<char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col--; });

   std::vector<double> values(nnz);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   if(values.size() != nnz)
   {
      THROWERROR("Invalid number of values");
   }

   if(rows.size() != nnz)
   {
      THROWERROR("Invalid number of rows");
   }

   if(cols.size() != nnz)
   {
      THROWERROR("Invalid number of columns");
   }

   return std::make_shared<MatrixConfig>(nrow, ncol, rows, cols, values, NoiseConfig(), isScarce);
}

std::shared_ptr<MatrixConfig> matrix_io::read_sparse_binary_bin(std::istream& in, bool isScarce)
{
   std::uint64_t nrow;
   std::uint64_t ncol;
   std::uint64_t nnz;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   std::vector<std::uint32_t> rows(nnz);
   in.read(reinterpret_cast<char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row--; });

   std::vector<std::uint32_t> cols(nnz);
   in.read(reinterpret_cast<char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col--; });

   return std::make_shared<MatrixConfig>(nrow, ncol, rows, cols, NoiseConfig(), isScarce);
}

// MatrixMarket format specification
// https://github.com/ExaScience/smurff/files/1398286/MMformat.pdf
std::shared_ptr<MatrixConfig> matrix_io::read_matrix_market(std::istream& in, bool isScarce)
{
   // Check that stream has MatrixMarket format data
   std::array<char, 15> matrixMarketArr;
   in.read(matrixMarketArr.data(), 14);
   std::string matrixMarketStr(matrixMarketArr.begin(), matrixMarketArr.end());
   if (matrixMarketStr != "%%MatrixMarket" && !std::isblank(in.get()))
   {
      std::stringstream ss;
      ss << "Cannot read MatrixMarket from input stream: ";
      ss << "the first 15 characters must be '%%MatrixMarket' followed by at least one blank";
      THROWERROR(ss.str());
   }

   // Parse MatrixMarket header
   std::string headerStr;
   std::getline(in, headerStr);
   std::stringstream headerStream(headerStr);

   std::string object;
   headerStream >> object;
   std::transform(object.begin(), object.end(), object.begin(), ::toupper);

   std::string format;
   headerStream >> format;
   std::transform(format.begin(), format.end(), format.begin(), ::toupper);

   std::string field;
   headerStream >> field;
   std::transform(field.begin(), field.end(), field.begin(), ::toupper);

   std::string symmetry;
   headerStream >> symmetry;
   std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::toupper);

   // Check object type
   if (object != MM_OBJ_MATRIX)
   {
      std::stringstream ss;
      ss << "Invalid MartrixMarket object type: expected 'matrix' but got '" << object << "'";
      THROWERROR(ss.str());
   }

   // Check field type
   if (field != MM_FLD_REAL && field != MM_FLD_PATTERN)
   {
      THROWERROR("Invalid MatrixMarket field type: only 'real' and 'pattern' field types are supported");
   }

   // Check symmetry type
   if (symmetry != MM_SYM_GENERAL)
   {
      THROWERROR("Invalid MatrixMarket symmetry type: only 'general' symmetry type is supported");
   }

   // Skip comments and empty lines
   while (in.peek() == '%' || in.peek() == '\n')
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

   // Read data
   if (format == MM_FMT_COORD)
   {
      std::uint64_t nrows;
      std::uint64_t ncols;
      std::uint64_t nnz;
      in >> nrows >> ncols >> nnz;

      if (in.fail())
      {
         THROWERROR("Could not get 'rows', 'cols', 'nnz' values for coordinate matrix format");
      }

      std::vector<std::uint32_t> rows(nnz);
      std::vector<std::uint32_t> cols(nnz);
      std::vector<double> vals(nnz);

      for (std::uint64_t i = 0; i < nnz; i++)
      {
         while (in.peek() == '%' || in.peek() == '\n')
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

         std::uint32_t row;
         std::uint32_t col;
         double val;

         if (field == MM_FLD_REAL)
         {
            in >> row >> col >> val;
         }
         else if (field == MM_FLD_PATTERN)
         {
            in >> row >> col;
            val = 1.0;
         }
         else
         {
            THROWERROR("Invalid MatrixMarket field type: coord format supports only 'real' and 'pattern' field types");
         }

         if (in.fail())
         {
            THROWERROR("Could not parse an entry line for coordinate matrix format");
         }

         rows[i] = row - 1;
         cols[i] = col - 1;
         vals[i] = val;
      }

      return std::make_shared<MatrixConfig>(nrows, ncols, rows, cols, vals, NoiseConfig(), isScarce);
   }
   else if (format == MM_FMT_ARRAY)
   {
      if (field != MM_FLD_REAL)
      {
         THROWERROR("Invalid MatrixMarket field type: array format supports only 'real' field type");
      }

      std::uint64_t nrows;
      std::uint64_t ncols;
      in >> nrows >> ncols;

      if (in.fail())
      {
         THROWERROR("Could not get 'rows', 'cols' values for array matrix format");
      }

      std::vector<double> vals(nrows * ncols);
      for (double& val : vals)
      {
         while (in.peek() == '%' || in.peek() == '\n')
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

         in >> val;
         if (in.fail())
         {
            THROWERROR("Could not parse an entry line for array matrix format");
         }
      }

      return std::make_shared<MatrixConfig>(nrows, ncols, vals, NoiseConfig());
   }
   else
   {
      std::stringstream ss;
      ss << "Invalid MatrixMarket format type: expected 'coordinate' or 'array' but got '" << format << "'";
      THROWERROR(ss.str());
   }
}

// ======================================================================================================

void matrix_io::write_matrix(const std::string& filename, std::shared_ptr<const MatrixConfig> matrixConfig)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         matrix_io::write_sparse_float64_bin(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::sbm:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         matrix_io::write_sparse_binary_bin(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::mtx:
      {
         std::ofstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         matrix_io::write_matrix_market(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::csv:
      {
         std::ofstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         matrix_io::write_dense_float64_csv(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::ddm:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         matrix_io::write_dense_float64_bin(fileStream, matrixConfig);
      }
      break;
   case matrix_io::MatrixType::none:
      {
         THROWERROR("Unknown matrix type");
      }
   default:
      {
         THROWERROR("Unknown matrix type");
      }
   }
}

void matrix_io::write_dense_float64_bin(std::ostream& out, std::shared_ptr<const MatrixConfig> matrixConfig)
{
   std::uint64_t nrow = matrixConfig->getNRow();
   std::uint64_t ncol = matrixConfig->getNCol();
   const std::vector<double>& values = matrixConfig->getValues();

   out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void matrix_io::write_dense_float64_csv(std::ostream& out, std::shared_ptr<const MatrixConfig> matrixConfig)
{
   //write rows and cols
   std::uint64_t nrow = matrixConfig->getNRow();
   std::uint64_t ncol = matrixConfig->getNCol();

   out << nrow << std::endl;
   out << ncol << std::endl;

   const std::vector<double>& values = matrixConfig->getValues();

   if(values.size() != nrow * ncol)
   {
      THROWERROR("invalid number of values");
   }

   //write values
   for(std::uint64_t i = 0; i < nrow; i++)
   {
      for(std::uint64_t j = 0; j < ncol; j++)
      {
         if(j == ncol - 1)
            out << values[j * nrow + i];
         else
            out << values[j * nrow + i] << ",";
      }
      out << std::endl;
   }
}

void matrix_io::write_sparse_float64_bin(std::ostream& out, std::shared_ptr<const MatrixConfig> matrixConfig)
{
   std::uint64_t nrow = matrixConfig->getNRow();
   std::uint64_t ncol = matrixConfig->getNCol();
   std::uint64_t nnz = matrixConfig->getNNZ();

   //get values copy
   std::vector<std::uint32_t> rows = matrixConfig->getRows();
   std::vector<std::uint32_t> cols = matrixConfig->getCols();
   std::vector<double> values = matrixConfig->getValues();

   //increment coordinates
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row++; });
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col++; });

   out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void matrix_io::write_sparse_binary_bin(std::ostream& out, std::shared_ptr<const MatrixConfig> matrixConfig)
{
   std::uint64_t nrow = matrixConfig->getNRow();
   std::uint64_t ncol = matrixConfig->getNCol();
   std::uint64_t nnz = matrixConfig->getNNZ();

   //get values copy
   std::vector<std::uint32_t> rows = matrixConfig->getRows();
   std::vector<std::uint32_t> cols = matrixConfig->getCols();

   //increment coordinates
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row++; });
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col++; });

   out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
}

// MatrixMarket format specification
// https://github.com/ExaScience/smurff/files/1398286/MMformat.pdf
void matrix_io::write_matrix_market(std::ostream& out, std::shared_ptr<const MatrixConfig> matrixConfig)
{
   out << "%%MatrixMarket ";
   out << MM_OBJ_MATRIX << " ";
   out << (matrixConfig->isDense() ? MM_FMT_ARRAY : MM_FMT_COORD) << " ";
   out << (matrixConfig->isBinary() ? MM_FLD_PATTERN : MM_FLD_REAL) << " ";
   out << MM_SYM_GENERAL << std::endl;

   if (matrixConfig->isDense())
   {
      out << matrixConfig->getNRow() << " ";
      out << matrixConfig->getNCol() << std::endl;
      for (const double& val : matrixConfig->getValues())
         out << val << std::endl;
   }
   else
   {
      out << matrixConfig->getNRow() << " ";
      out << matrixConfig->getNCol() << " ";
      out << matrixConfig->getNNZ() << std::endl;
      for (std::uint64_t i = 0; i < matrixConfig->getNNZ(); i++)
      {
         const std::uint32_t& row = matrixConfig->getRows()[i] + 1;
         const std::uint32_t& col = matrixConfig->getCols()[i] + 1;
         if (matrixConfig->isBinary())
         {
            out << row << " " << col << std::endl;
         }
         else
         {
            const double& val = matrixConfig->getValues()[i];
            out << row << " " << col << " " << val << std::endl;
         }
      }
   }
}

// ======================================================================================================

void matrix_io::eigen::read_matrix(const std::string& filename, Vector& V)
{
   Matrix X;
   matrix_io::eigen::read_matrix(filename, X);
   V = X; // this will fail if X has more than one column
}


void matrix_io::eigen::read_dense_float64_bin(std::istream& in, Matrix& X)
{
   std::uint64_t nrow = 0;
   std::uint64_t ncol = 0;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));

   Eigen::MatrixXd X_as_double(nrow, ncol);
   in.read(reinterpret_cast<char*>(X_as_double.data()), nrow * ncol * sizeof(typename Eigen::MatrixXd::Scalar));

   X = X_as_double.cast<Matrix::Scalar>();
}

void matrix_io::eigen::read_dense_float64_csv(std::istream& in, Matrix& X)
{
   std::stringstream ss;
   std::string line;

   // rows and cols
   getline(in, line);
   ss.clear();
   ss << line;
   std::uint64_t nrow;
   ss >> nrow;

   getline(in, line);
   ss.clear();
   ss << line;
   std::uint64_t ncol;
   ss >> ncol;

   X.resize(nrow, ncol);

   std::uint64_t row = 0;
   std::uint64_t col = 0;

   while(getline(in, line) && row < nrow)
   {
      col = 0;

      std::stringstream lineStream(line);
      std::string cell;

      while (std::getline(lineStream, cell, ',') && col < ncol)
      {
         X(row, col++) = stod(cell);
      }

      row++;
   }

   if(row != nrow)
   {
      THROWERROR("invalid number of rows");
   }

   if(col != ncol)
   {
      THROWERROR("invalid number of columns");
   }
}


// MatrixMarket format specification
// https://github.com/ExaScience/smurff/files/1398286/MMformat.pdf
void matrix_io::eigen::read_matrix_market(std::istream& in, Matrix& X)
{
   // Check that stream has MatrixMarket format data
   std::array<char, 15> matrixMarketArr;
   in.read(matrixMarketArr.data(), 14);
   std::string matrixMarketStr(matrixMarketArr.begin(), matrixMarketArr.end());
   if (matrixMarketStr != "%%MatrixMarket" && !std::isblank(in.get()))
   {
      std::stringstream ss;
      ss << "Cannot read MatrixMarket from input stream: ";
      ss << "the first 15 characters must be '%%MatrixMarket' followed by at least one blank";
      THROWERROR(ss.str());
   }

   // Parse MatrixMarket header
   std::string headerStr;
   std::getline(in, headerStr);
   std::stringstream headerStream(headerStr);

   std::string object;
   headerStream >> object;
   std::transform(object.begin(), object.end(), object.begin(), ::toupper);

   std::string format;
   headerStream >> format;
   std::transform(format.begin(), format.end(), format.begin(), ::toupper);

   std::string field;
   headerStream >> field;
   std::transform(field.begin(), field.end(), field.begin(), ::toupper);

   std::string symmetry;
   headerStream >> symmetry;
   std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::toupper);

   // Check object type
   if (object != MM_OBJ_MATRIX)
   {
      std::stringstream ss;
      ss << "Invalid MartrixMarket object type: expected 'matrix' but got '" << object << "'";
      THROWERROR(ss.str());
   }

   // Check field type
   if (field != MM_FLD_REAL)
   {
      THROWERROR("Invalid MatrixMarket field type: only 'real' field type is supported");
   }

   // Check symmetry type
   if (symmetry != MM_SYM_GENERAL)
   {
      THROWERROR("Invalid MatrixMarket symmetry type: only 'general' symmetry type is supported");
   }

   // Skip comments and empty lines
   while (in.peek() == '%' || in.peek() == '\n')
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

   if (format != MM_FMT_ARRAY)
   {
      THROWERROR("Cannot read a sparse matrix as Matrix");
   }

   std::uint64_t nrows;
   std::uint64_t ncols;
   in >> nrows >> ncols;

   if (in.fail())
   {
      THROWERROR("Could not get 'rows', 'cols' values for array matrix format");
   }

   X.resize(nrows, ncols);

   for (std::uint64_t col = 0; col < ncols; col++)
   {
      for (std::uint64_t row = 0; row < nrows; row++)
      {
         while (in.peek() == '%' || in.peek() == '\n')
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

         double val;
         in >> val;
         if (in.fail())
         {
            THROWERROR("Could not parse an entry line for array matrix format");
         }

         X(row, col) = val;
      }
   }
}

void matrix_io::eigen::read_matrix(const std::string& filename, Matrix& X)
{
   matrix_io::MatrixType matrixType = matrix_io::ExtensionToMatrixType(filename);

   THROWERROR_FILE_NOT_EXIST(filename);

   switch (matrixType)
   {
   case matrix_io::MatrixType::sdm:
      {
         THROWERROR("Invalid matrix type");
      }
   case matrix_io::MatrixType::sbm:
      {
         THROWERROR("Invalid matrix type");
      }
   case matrix_io::MatrixType::mtx:
      {
         std::ifstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         read_matrix_market(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::csv:
      {
         std::ifstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         read_dense_float64_csv(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::ddm:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         read_dense_float64_bin(fileStream, X);
      }
      break;
   case matrix_io::MatrixType::none:
      {
         THROWERROR("Unknown matrix type");
      }
   default:
      {
         THROWERROR("Unknown matrix type");
      }
   }
}

void matrix_io::eigen::read_matrix(const std::string& filename, SparseMatrix& X)
{
   auto ptr = matrix_io::read_matrix(filename, false);
   X = matrix_utils::sparse_to_eigen(*ptr);
}

// ======================================================================================================

void matrix_io::eigen::write_matrix(const std::string& filename, const Matrix& X)
{
   matrix_io::write_matrix(filename, matrix_utils::eigen_to_dense(X));
   H5Easy::File file(filename + ".h5", H5Easy::File::Overwrite);
   H5Easy::dump(file, "/path/to/A", X);
}

void matrix_io::eigen::write_matrix(const std::string& filename, const SparseMatrix& X)
{
   matrix_io::write_matrix(filename, matrix_utils::eigen_to_sparse(X));
}
} // end namespace smurff
