#pragma once
// clang_long_double.hpp: nonconstexpr implementation of IEEE-754 long double
// manipulators
//
// Copyright (C) 2017-2022 Stillwater Supercomputing, Inc.
//
// This file is part of the universal numbers project, which is released under
// an MIT Open Source license.

#if defined(__clang__)
/* Clang/LLVM. ---------------------------------------------- */

namespace sw {
namespace universal {

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// compiler specific long double IEEE floating point

// __arm__ which is defined for 32bit arm, and 32bit arm only.
// __aarch64__ which is defined for 64bit arm, and 64bit arm only.

#if defined(__aarch64__)
union long_double_decoder {
  long_double_decoder() : ld{0.0l} {}
  long_double_decoder(long double _ld) : ld{_ld} {}
  long double ld;
  struct {
    uint64_t fraction : 112;
    uint64_t exponent : 15;
    uint64_t sign : 1;
  } parts;
};
#else
union long_double_decoder {
  long_double_decoder() : ld{0.0l} {}
  long_double_decoder(long double _ld) : ld{_ld} {}
  long double ld;
  struct {
    uint64_t fraction : 63;
    uint64_t bit63 : 1;
    uint64_t exponent : 15;
    uint64_t sign : 1;
  } parts;
};
#endif // __aarch64__

// extract the fields of a native C++ long double
inline void extractFields(long double value, bool &s, uint64_t &rawExponentBits,
                          uint64_t &rawFractionBits) {
  long_double_decoder decoder;
  decoder.ld = value;
  s = decoder.parts.sign == 1 ? true : false;
  rawExponentBits = decoder.parts.exponent;
  rawFractionBits = decoder.parts.fraction;
}

// ieee_components returns a tuple of sign, exponent, and fraction.
inline std::tuple<bool, int, std::uint64_t> ieee_components(long double fp) {
  static_assert(
      std::numeric_limits<double>::is_iec559,
      "This function only works when double complies with IEC 559 (IEEE 754)");
  static_assert(sizeof(long double) == 16,
                "This function only works when double is 80 bit.");

  long_double_decoder dd{fp}; // initializes the first member of the union
  // Reading inactive union parts is forbidden in constexpr :-(
  return std::make_tuple<bool, int, std::uint64_t>(
      static_cast<bool>(dd.parts.sign), static_cast<int>(dd.parts.exponent),
      static_cast<std::uint64_t>(dd.parts.fraction));
}

// specialization for IEEE long double precision floats
inline std::string to_base2_scientific(long double number) {
  std::stringstream s;
  long_double_decoder decoder;
  decoder.ld = number;
  s << (decoder.parts.sign == 1 ? "-" : "+") << "1.";
  uint64_t mask = (uint64_t(1) << 63);
  for (int i = 63; i >= 0; --i) {
    s << ((decoder.parts.fraction & mask) ? '1' : '0');
    mask >>= 1;
  }
  s << "e" << std::showpos
    << (static_cast<int>(decoder.parts.exponent) - 16383);
  return s.str();
}

// generate a binary string for a native double precision IEEE floating point
inline std::string to_hex(long double number) {
  std::stringstream s;
  long_double_decoder decoder;
  decoder.ld = number;
  s << (decoder.parts.sign ? '1' : '0') << '.' << std::hex
    << int(decoder.parts.exponent) << '.' << decoder.parts.fraction;
  return s.str();
}

// generate a binary string for a native double precision IEEE floating point
inline std::string to_binary(long double number, bool bNibbleMarker = false) {
  std::stringstream s;
  long_double_decoder decoder;
  decoder.ld = number;

  s << "0b";
  // print sign bit
  s << (decoder.parts.sign ? '1' : '0') << '.';

  // print exponent bits
  {
    uint64_t mask = 0x4000;
    for (int i = 14; i >= 0; --i) {
      s << ((decoder.parts.exponent & mask) ? '1' : '0');
      if (bNibbleMarker && i != 0 && (i % 4) == 0)
        s << '\'';
      mask >>= 1;
    }
  }

  s << '.';

  // print fraction bits
  uint64_t mask = (uint64_t(1) << 62);
  for (int i = 62; i >= 0; --i) {
    s << ((decoder.parts.fraction & mask) ? '1' : '0');
    if (bNibbleMarker && i != 0 && (i % 4) == 0)
      s << '\'';
    mask >>= 1;
  }

  return s.str();
}

// return in triple form (+, scale, fraction)
inline std::string to_triple(long double number) {
  std::stringstream s;
  long_double_decoder decoder;
  decoder.ld = number;

  // print sign bit
  s << '(' << (decoder.parts.sign ? '-' : '+') << ',';

  // exponent
  // the exponent value used in the arithmetic is the exponent shifted by a bias
  // for the IEEE 754 binary32 case, an exponent value of 127 represents the
  // actual zero (i.e. for 2^(e ¿ 127) to be one, e must be 127). Exponents
  // range from ¿126 to +127 because exponents of ¿127 (all 0s) and +128 (all
  // 1s) are reserved for special numbers.
  if (decoder.parts.exponent == 0) {
    s << "exp=0,";
  } else if (decoder.parts.exponent == 0xFF) {
    s << "exp=1, ";
  }
  int scale = int(decoder.parts.exponent) - 16383;
  s << scale << ',';

  // print fraction bits
  s << (decoder.parts.bit63 ? '1' : '0');
  uint64_t mask = (uint64_t(1) << 62);
  for (int i = 62; i >= 0; --i) {
    s << ((decoder.parts.fraction & mask) ? '1' : '0');
    mask >>= 1;
  }

  s << ')';
  return s.str();
}

// generate a color coded binary string for a native double precision IEEE
// floating point
inline std::string color_print(long double number) {
  std::stringstream s;
  long_double_decoder decoder;
  decoder.ld = number;

  Color red(ColorCode::FG_RED);
  Color yellow(ColorCode::FG_YELLOW);
  Color blue(ColorCode::FG_BLUE);
  Color magenta(ColorCode::FG_MAGENTA);
  Color cyan(ColorCode::FG_CYAN);
  Color white(ColorCode::FG_WHITE);
  Color def(ColorCode::FG_DEFAULT);

  // print prefix
  s << yellow << "0b";

  // print sign bit
  s << red << (decoder.parts.sign ? '1' : '0') << '.';

  // print exponent bits
  {
    uint64_t mask = 0x8000;
    for (int i = 15; i >= 0; --i) {
      s << cyan << ((decoder.parts.exponent & mask) ? '1' : '0');
      if (i > 0 && i % 4 == 0)
        s << cyan << '\'';
      mask >>= 1;
    }
  }

  s << '.';

  // print fraction bits
  s << magenta << (decoder.parts.bit63 ? '1' : '0');
  uint64_t mask = (uint64_t(1) << 61);
  for (int i = 61; i >= 0; --i) {
    s << magenta << ((decoder.parts.fraction & mask) ? '1' : '0');
    if (i > 0 && i % 4 == 0)
      s << magenta << '\'';
    mask >>= 1;
  }

  s << def;
  return s.str();
}

#ifdef CPLUSPLUS_17
inline void extract_fp_components(long double fp, bool &_sign, int &_exponent,
                                  long double &_fr,
                                  unsigned long long &_fraction) {
  if constexpr (std::numeric_limits<long double>::digits <= 64) {
    if constexpr (sizeof(long double) == 8) { // it is just a double
      _sign = fp < 0.0 ? true : false;
      _fr = frexp(double(fp), &_exponent);
      _fraction =
          uint64_t(0x000FFFFFFFFFFFFFull) & reinterpret_cast<uint64_t &>(_fr);
    } else if constexpr (sizeof(long double) == 16 &&
                         std::numeric_limits<long double>::digits <= 64) {
      _sign = fp < 0.0 ? true : false;
      _fr = frexpl(fp, &_exponent);
      _fraction = uint64_t(0x7FFFFFFFFFFFFFFFull) &
                  reinterpret_cast<uint64_t &>(
                      _fr); // 80bit extended format only has 63bits of fraction
    }
  } else if constexpr (std::numeric_limits<long double>::digits == 113) {
    std::cerr << "numeric_limits<long double>::digits = "
              << std::numeric_limits<long double>::digits
              << " currently unsupported\n";
  }
}
#else
inline void extract_fp_components(long double fp, bool &_sign, int &_exponent,
                                  long double &_fr,
                                  unsigned long long &_fraction) {
  if (std::numeric_limits<long double>::digits <= 64) {
    if (sizeof(long double) == 8) { // it is just a double
      _sign = fp < 0.0 ? true : false;
      _fr = frexp(double(fp), &_exponent);
      _fraction =
          uint64_t(0x000FFFFFFFFFFFFFull) & reinterpret_cast<uint64_t &>(_fr);
    } else if (sizeof(long double) == 16 &&
               std::numeric_limits<long double>::digits <= 64) {
      _sign = fp < 0.0 ? true : false;
      _fr = frexpl(fp, &_exponent);
      _fraction = uint64_t(0x7FFFFFFFFFFFFFFFull) &
                  reinterpret_cast<uint64_t &>(
                      _fr); // 80bit extended format only has 63bits of fraction
    }
  } else {
    std::cerr << "numeric_limits<long double>::digits = "
              << std::numeric_limits<long double>::digits
              << " currently unsupported\n";
  }
}
#endif

} // namespace universal
} // namespace sw

#endif // CLANG/LLVM
