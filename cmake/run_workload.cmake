if(NOT DEFINED NOCDAS_MODEL)
    set(NOCDAS_MODEL "$ENV{MODEL}")
endif()
if(NOT DEFINED NOCDAS_EXP)
    set(NOCDAS_EXP "$ENV{EXP}")
endif()
if(NOT NOCDAS_MODEL MATCHES "^[A-Za-z0-9_]+$")
    message(FATAL_ERROR "MODEL is required and must contain only letters, digits, and underscores")
endif()
if(NOT NOCDAS_EXP MATCHES "^[0-9]+$")
    message(FATAL_ERROR "EXP is required and must be a non-negative integer")
endif()
if(NOT NOCDAS_MODE MATCHES "^(baseline|cnoc)$")
    message(FATAL_ERROR "Invalid simulator mode: ${NOCDAS_MODE}")
endif()

set(NOCDAS_INPUT_DIR "${NOCDAS_SOURCE_DIR}/src/input/${NOCDAS_MODEL}")
set(NOCDAS_MODEL_FILE "${NOCDAS_INPUT_DIR}/lm_transformer_${NOCDAS_MODEL}_exp${NOCDAS_EXP}.txt")
set(NOCDAS_WEIGHT_FILE "${NOCDAS_INPUT_DIR}/lm_weight_${NOCDAS_MODEL}_exp${NOCDAS_EXP}.txt")
set(NOCDAS_INPUT_FILE "${NOCDAS_INPUT_DIR}/lm_input_${NOCDAS_MODEL}_exp${NOCDAS_EXP}.txt")
foreach(file IN ITEMS "${NOCDAS_MODEL_FILE}" "${NOCDAS_WEIGHT_FILE}" "${NOCDAS_INPUT_FILE}")
    if(NOT EXISTS "${file}")
        message(FATAL_ERROR "Workload file not found: ${file}")
    endif()
endforeach()

set(NOCDAS_LOG_DIR "${NOCDAS_SOURCE_DIR}/logs/${NOCDAS_MODE}")
file(MAKE_DIRECTORY "${NOCDAS_LOG_DIR}")
set(NOCDAS_LOG_FILE "${NOCDAS_LOG_DIR}/${NOCDAS_MODEL}_exp${NOCDAS_EXP}.log")

execute_process(
    COMMAND /bin/bash -o pipefail -c
        "\"${NOCDAS_BINARY}\" -NNmodel \"${NOCDAS_MODEL_FILE}\" -NNweight \"${NOCDAS_WEIGHT_FILE}\" -NNinput \"${NOCDAS_INPUT_FILE}\" 2>&1 | tee \"${NOCDAS_LOG_FILE}\""
    WORKING_DIRECTORY "${NOCDAS_SOURCE_DIR}"
    RESULT_VARIABLE NOCDAS_RESULT
)
if(NOT NOCDAS_RESULT EQUAL 0)
    message(FATAL_ERROR "Workload failed with exit code ${NOCDAS_RESULT}; see ${NOCDAS_LOG_FILE}")
endif()
message(STATUS "${NOCDAS_MODE} ${NOCDAS_MODEL} Exp.${NOCDAS_EXP} log: ${NOCDAS_LOG_FILE}")
