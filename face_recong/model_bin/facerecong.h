// Bundle API auto-generated header file. Do not edit!
// Glow Tools version: 2021-06-10 (066c3fca1)

#ifndef _GLOW_BUNDLE_FACERECONG_H
#define _GLOW_BUNDLE_FACERECONG_H

#include <stdint.h>

// ---------------------------------------------------------------
//                       Common definitions
// ---------------------------------------------------------------
#ifndef _GLOW_BUNDLE_COMMON_DEFS
#define _GLOW_BUNDLE_COMMON_DEFS

// Glow bundle error code for correct execution.
#define GLOW_SUCCESS 0

// Memory alignment definition with given alignment size
// for static allocation of memory.
#define GLOW_MEM_ALIGN(size)  __attribute__((aligned(size)))

// Macro function to get the absolute address of a
// placeholder using the base address of the mutable
// weight buffer and placeholder offset definition.
#define GLOW_GET_ADDR(mutableBaseAddr, placeholderOff)  (((uint8_t*)(mutableBaseAddr)) + placeholderOff)

#endif

// ---------------------------------------------------------------
//                          Bundle API
// ---------------------------------------------------------------
// Model name: "facerecong"
// Total data size: 190080 (bytes)
// Placeholders:
//
//   Name: "input_1"
//   Type: float<1 x 1 x 56 x 56>
//   Size: 3136 (elements)
//   Size: 12544 (bytes)
//   Offset: 0 (bytes)
//
//   Name: "A124"
//   Type: float<1 x 64>
//   Size: 64 (elements)
//   Size: 256 (bytes)
//   Offset: 12544 (bytes)
//
// NOTE: Placeholders are allocated within the "mutableWeight"
// buffer and are identified using an offset relative to base.
// ---------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// Placeholder address offsets within mutable buffer (bytes).
#define FACERECONG_input_1  0
#define FACERECONG_A124     12544

// Memory sizes (bytes).
#define FACERECONG_CONSTANT_MEM_SIZE     139648
#define FACERECONG_MUTABLE_MEM_SIZE      12800
#define FACERECONG_ACTIVATIONS_MEM_SIZE  37632

// Memory alignment (bytes).
#define FACERECONG_MEM_ALIGN  64

// Bundle entry point (inference function). Returns 0
// for correct execution or some error code otherwise.
int facerecong(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations);

#ifdef __cplusplus
}
#endif
#endif
