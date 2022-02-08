################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c \
../Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.c 

OBJS += \
./Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.o \
./Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.o 

C_DEPS += \
./Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.d \
./Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/%.o: ../Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/%.c Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O3 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-CenyML_Library-2f-cpuSequential-2f-evaluationMetrics

clean-Core-2f-Src-2f-CenyML_Library-2f-cpuSequential-2f-evaluationMetrics:
	-$(RM) ./Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.d ./Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.o ./Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.d ./Core/Src/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.o

.PHONY: clean-Core-2f-Src-2f-CenyML_Library-2f-cpuSequential-2f-evaluationMetrics

