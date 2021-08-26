// Bundle API auto-generated header file. Do not edit!
// Glow Tools version: 2021-06-10 (066c3fca1)

#ifndef _GLOW_BUNDLE_LT_FLOOR_H
#define _GLOW_BUNDLE_LT_FLOOR_H

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
// Model name: "lt_floor"
// Total data size: 144320 (bytes)
// Placeholders:
//
//   Name: "input_data"
//   Type: float<1 x 1 x 120 x 120>
//   Size: 14400 (elements)
//   Size: 57600 (bytes)
//   Offset: 0 (bytes)
//
//   Name: "probe"
//   Type: float<1 x 5 x 7 x 7>
//   Size: 245 (elements)
//   Size: 980 (bytes)
//   Offset: 57600 (bytes)
//
//   Name: "pred_x"
//   Type: float<1 x 5 x 7 x 7>
//   Size: 245 (elements)
//   Size: 980 (bytes)
//   Offset: 58624 (bytes)
//
//   Name: "pred_y"
//   Type: float<1 x 5 x 7 x 7>
//   Size: 245 (elements)
//   Size: 980 (bytes)
//   Offset: 59648 (bytes)
//
// NOTE: Placeholders are allocated within the "mutableWeight"
// buffer and are identified using an offset relative to base.
// ---------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// Placeholder address offsets within mutable buffer (bytes).
#define LT_FLOOR_input_data  0
#define LT_FLOOR_probe       57600
#define LT_FLOOR_pred_x      58624
#define LT_FLOOR_pred_y      59648

// Memory sizes (bytes).
#define LT_FLOOR_CONSTANT_MEM_SIZE     40448
#define LT_FLOOR_MUTABLE_MEM_SIZE      60672
#define LT_FLOOR_ACTIVATIONS_MEM_SIZE  43200

// Memory alignment (bytes).
#define LT_FLOOR_MEM_ALIGN  64

// Bundle entry point (inference function). Returns 0
// for correct execution or some error code otherwise.
int lt_floor(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations);

#ifdef __cplusplus
}
#endif
#endif
