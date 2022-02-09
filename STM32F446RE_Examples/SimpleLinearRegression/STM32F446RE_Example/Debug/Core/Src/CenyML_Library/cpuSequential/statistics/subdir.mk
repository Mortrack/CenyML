################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/CenyML_Library/cpuSequential/statistics/CenyMLstatistics.c 

OBJS += \
./Core/Src/CenyML_Library/cpuSequential/statistics/CenyMLstatistics.o 

C_DEPS += \
./Core/Src/CenyML_Library/cpuSequential/statistics/CenyMLstatistics.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/CenyML_Library/cpuSequential/statistics/%.o: ../Core/Src/CenyML_Library/cpuSequential/statistics/%.c Core/Src/CenyML_Library/cpuSequential/statistics/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O3 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-CenyML_Library-2f-cpuSequential-2f-statistics

clean-Core-2f-Src-2f-CenyML_Library-2f-cpuSequential-2f-statistics:
	-$(RM) ./Core/Src/CenyML_Library/cpuSequential/statistics/CenyMLstatistics.d ./Core/Src/CenyML_Library/cpuSequential/statistics/CenyMLstatistics.o

.PHONY: clean-Core-2f-Src-2f-CenyML_Library-2f-cpuSequential-2f-statistics

