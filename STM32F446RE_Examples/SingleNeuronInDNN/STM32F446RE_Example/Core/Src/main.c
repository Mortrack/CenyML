/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stdio.h"
#include "CenyML_Library/cpuSequential/machineLearning/CenyMLdeepLearning.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
RTC_HandleTypeDef hrtc;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_RTC_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/*
 * The _write() function is used to forward all the printf commands to the ITM,
 * which is the STM32 IDE component that manages the data console of such IDE.
 * However, in order for you to see any printed messaged in the SWV ITM Data
 * Console tool, all your printf strings will have to end with a new line (\n).
 *
 * return int len
 *
 * @author Controllers Tech
 * CREATION DATE: JANUARY 07, 2022.
 * LAST UPDATE: N/A/
 */
int _write(int file, char *ptr, int len) {
	for (int i=0; i<len; i++) {
		ITM_SendChar((*ptr++));
	}
	return len;
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_RTC_Init();
  /* USER CODE BEGIN 2 */
  	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	struct singleNeuronDnnStruct neuron1; // We create a singleNeuronDnnStruct structure variable to manage the data input and output data of the single neuron in DNN that will be created.
	neuron1.n = 1990; // This variable will contain the number of samples that the input matrix is expected to have.
	neuron1.m = 1; // This variable will contain the number of features (independent variables) that the input matrix is expected to have.
	neuron1.p = 1; // This variable will contain the number of outputs that the output matrix is expected to have.
	neuron1.isInitial_w = 1; // This variable will indicate whether or not initial values will be given by the implementer (with value of 1) or if random ones are going to be used (with value of 0).
	neuron1.w_first = (double *) malloc((neuron1.m+1)*sizeof(double)); // We allocate the memory required for the variable "neuron1.w_first", which will store the initial coefficient values of the neuron to be created.
	neuron1.w_first[0] = 0; // We define the customized desired value for the bias of the neuron to be created.
	neuron1.w_first[1] = 0; // We define the customized desired value for the weight_1 value of the neuron to be created.
	neuron1.isClassification = 0; // This variable will indicate whether or not it is desired that the neuron considers the input data for a classification (with a vlaue of 1) or a regression problem (with a value of 0).
	//neuron1.threshold = 0.5; // This variable will be used to store the desired threshold value to be used in classification problems by the neuron to be created.
	//neuron1.desiredValueForGroup1 = 1; // This variable will be used to store the label to be used for the group 1 in classification problems by the neuron to be created.
	//neuron1.desiredValueForGroup2 = -1; // This variable will be used to store the label to be used for the group 2 in classification problems by the neuron to be created.
	neuron1.activationFunctionToBeUsed = 3; // This variable tells the neuron what activation function to use (see the commented documentation in the function "getSingleNeuronDNN()" for more details).
	neuron1.learningRate = 0.0000001; // This variable stores the desired learning rate for the neuron to be created.
	neuron1.stopAboveThisAccuracy = 0.99; // The value of this variable is used as a stop function for the single neuron in DNN learning proccess.
	neuron1.maxEpochs = 18000; // This variable stores the desired value for the maximum permitted epochs for the training process of the neuron.
	neuron1.isReportLearningProgress = 1; // The value of this variable tells the neuron if it is desired that it reports its learning progress (with a value of 1) or not (with a value of 0).
	neuron1.reportEachSpecifiedEpochs = neuron1.maxEpochs / 10; // This variable tells the neuron that it has to report each several times, which is defined by the value contained in this variable.
	neuron1.Y = (double *) malloc(neuron1.n*neuron1.p*sizeof(double)); // Allocate the memory required for the variable "neuron1.Y", which will contain the real output data of the system under study.
	neuron1.X = (double *) malloc(neuron1.n*neuron1.m*sizeof(double)); // Allocate the memory required for the variable "neuron1.X", which will contain the input data of the system under study.
	double X[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; // This variable contains the input data of the system under study for when n=10.
	double Y[] = {18, 26, 34, 42, 50, 58, 66, 74, 82, 90}; // This variable contains the expected/real output data of the system under study for when n=10.
	neuron1.w_best = (double *) malloc((neuron1.m+1)*sizeof(double)); // We allocate the memory required for the variable "neuron1.w_best", which will store the best coefficient values identified by the neuron to be created, after its training process.
	neuron1.w_new = (double *) malloc((neuron1.m+1)*sizeof(double)); // We allocate the memory required for the variable "neuron1.w_new", which will store the last coefficient values identified by the neuron to be created, after its training process.
	double b_ideal[2]; // This variable will be used to contain the ideal coefficient values that the model to be trained should give.
	b_ideal[0] = 10;
	b_ideal[1] = 0.8;

	long int startingTime; // This variable will be used to store the starting timestamp in milliseconds of a certain process in which the time will be counted.
	long int elapsedTime; // This variable will be used to store the total time in milliseconds of a certain process in which the time will be counted.
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	printf("----------------------------------------------------------------------\n");
	printf("----------------------------------------------------------------------\n");
	printf("Initializing the output and input data with %d samples for each of the %d columns (total samples = %d) each...\n", neuron1.n, neuron1.m, (neuron1.n*neuron1.m));
	startingTime = HAL_GetTick(); // We obtain the reference time to count the elapsed time to initialize the input data to be used.
	// Create the output (neuron1.Y) and input (neuron1.X) data with the same rows as in the reference .csv file and their corresponding number of columns.
	for (int currentIteration=0; currentIteration<(neuron1.n/10); currentIteration++) {
		for (int currentRow=0; currentRow<10; currentRow++) {
			neuron1.Y[currentRow + currentIteration*10] = Y[currentRow];
			neuron1.X[currentRow + currentIteration*10] = X[currentRow];
		}
	}
	elapsedTime = HAL_GetTick() - startingTime; // We obtain the elapsed time to initialize the input data to be used.
	printf("Output and input data initialization elapsed %ld milliseconds.\n\n", elapsedTime);


	// ------------------------- DATA MODELING ----------------------- //
	printf("Initializing CenyML single neuron in Deep Neural Network algorithm ...\n");
	startingTime = HAL_GetTick(); // We obtain the reference time to count the elapsed time to apply the single neuron in Deep Neural Network with the input data (neuron1.X).
	// We apply the single neuron in Deep Neural Network algorithm with respect to the input matrix "neuron1.X" and the result is stored in the memory location of the pointer "b".
	getSingleNeuronDNN(&neuron1);
	elapsedTime = HAL_GetTick() - startingTime; // We obtain the elapsed time to apply the single neuron in Deep Neural Network with the input data (neuron1.X).
	printf("CenyML single neuron in Deep Neural Network algorithm elapsed %ld milliseconds.\n\n", elapsedTime);
	printf("----------------------------------------------------------------------\n");
	printf("----------------------------------------------------------------------\n");


	HAL_Delay(1000); // 1000ms delay.
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSI|RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.LSEState = RCC_LSE_ON;
  RCC_OscInitStruct.LSIState = RCC_LSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief RTC Initialization Function
  * @param None
  * @retval None
  */
static void MX_RTC_Init(void)
{

  /* USER CODE BEGIN RTC_Init 0 */

  /* USER CODE END RTC_Init 0 */

  RTC_TimeTypeDef sTime = {0};
  RTC_DateTypeDef sDate = {0};

  /* USER CODE BEGIN RTC_Init 1 */

  /* USER CODE END RTC_Init 1 */
  /** Initialize RTC Only
  */
  hrtc.Instance = RTC;
  hrtc.Init.HourFormat = RTC_HOURFORMAT_24;
  hrtc.Init.AsynchPrediv = 127;
  hrtc.Init.SynchPrediv = 255;
  hrtc.Init.OutPut = RTC_OUTPUT_DISABLE;
  hrtc.Init.OutPutPolarity = RTC_OUTPUT_POLARITY_HIGH;
  hrtc.Init.OutPutType = RTC_OUTPUT_TYPE_OPENDRAIN;
  if (HAL_RTC_Init(&hrtc) != HAL_OK)
  {
    Error_Handler();
  }

  /* USER CODE BEGIN Check_RTC_BKUP */

  /* USER CODE END Check_RTC_BKUP */

  /** Initialize RTC and set the Time and Date
  */
  sTime.Hours = 0x0;
  sTime.Minutes = 0x0;
  sTime.Seconds = 0x0;
  sTime.DayLightSaving = RTC_DAYLIGHTSAVING_NONE;
  sTime.StoreOperation = RTC_STOREOPERATION_RESET;
  if (HAL_RTC_SetTime(&hrtc, &sTime, RTC_FORMAT_BCD) != HAL_OK)
  {
    Error_Handler();
  }
  sDate.WeekDay = RTC_WEEKDAY_MONDAY;
  sDate.Month = RTC_MONTH_JANUARY;
  sDate.Date = 0x1;
  sDate.Year = 0x0;

  if (HAL_RTC_SetDate(&hrtc, &sDate, RTC_FORMAT_BCD) != HAL_OK)
  {
    Error_Handler();
  }
  /** Enable the TimeStamp
  */
  if (HAL_RTCEx_SetTimeStamp(&hrtc, RTC_TIMESTAMPEDGE_RISING, RTC_TIMESTAMPPIN_DEFAULT) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN RTC_Init 2 */

  /* USER CODE END RTC_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
