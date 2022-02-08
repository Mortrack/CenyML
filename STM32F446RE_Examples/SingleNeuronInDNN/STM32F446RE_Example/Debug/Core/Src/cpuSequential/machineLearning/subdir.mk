################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/cpuSequential/machineLearning/CenyMLclassification.c \
../Core/Src/cpuSequential/machineLearning/CenyMLdeepLearning.c \
../Core/Src/cpuSequential/machineLearning/CenyMLregression.c 

OBJS += \
./Core/Src/cpuSequential/machineLearning/CenyMLclassification.o \
./Core/Src/cpuSequential/machineLearning/CenyMLdeepLearning.o \
./Core/Src/cpuSequential/machineLearning/CenyMLregression.o 

C_DEPS += \
./Core/Src/cpuSequential/machineLearning/CenyMLclassification.d \
./Core/Src/cpuSequential/machineLearning/CenyMLdeepLearning.d \
./Core/Src/cpuSequential/machineLearning/CenyMLregression.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/cpuSequential/machineLearning/%.o: ../Core/Src/cpuSequential/machineLearning/%.c Core/Src/cpuSequential/machineLearning/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-cpuSequential-2f-machineLearning

clean-Core-2f-Src-2f-cpuSequential-2f-machineLearning:
	-$(RM) ./Core/Src/cpuSequential/machineLearning/CenyMLclassification.d ./Core/Src/cpuSequential/machineLearning/CenyMLclassification.o ./Core/Src/cpuSequential/machineLearning/CenyMLdeepLearning.d ./Core/Src/cpuSequential/machineLearning/CenyMLdeepLearning.o ./Core/Src/cpuSequential/machineLearning/CenyMLregression.d ./Core/Src/cpuSequential/machineLearning/CenyMLregression.o

.PHONY: clean-Core-2f-Src-2f-cpuSequential-2f-machineLearning

