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
#include "CenyML_Library/cpuSequential/machineLearning/CenyMLregression.h"
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
  int n = 3000; // This variable will contain the number of samples that the system under study will have.
  int m = 2; // This variable will contain the number of features (independent variables) that the input matrix is expected to have.
  int p = 1; // This variable will contain the number of outputs that the output matrix is expected to have.
  int N = 2; // This variable will contain the value that is desired for the order of degree of the machine learning model to be trained.
  int n_database = 100; // This variable will define the number of samples that the reference .csv file (reference input data) contains.
  double databaseX[] = {10, 10, 10, 20, 10, 30, 10, 40, 10, 50, 10, 60, 10, 70, 10, 80, 10, 90, 10, 100,
  	  	  				20, 10, 20, 20, 20, 30, 20, 40, 20, 50, 20, 60, 20, 70, 20, 80, 20, 90, 20, 100,
						30, 10, 30, 20, 30, 30, 30, 40, 30, 50, 30, 60, 30, 70, 30, 80, 30, 90, 30, 100,
						40, 10, 40, 20, 40, 30, 40, 40, 40, 50, 40, 60, 40, 70, 40, 80, 40, 90, 40, 100,
						50, 10, 50, 20, 50, 30, 50, 40, 50, 50, 50, 60, 50, 70, 50, 80, 50, 90, 50, 100,
						60, 10, 60, 20, 60, 30, 60, 40, 60, 50, 60, 60, 60, 70, 60, 80, 60, 90, 60, 100,
						70, 10, 70, 20, 70, 30, 70, 40, 70, 50, 70, 60, 70, 70, 70, 80, 70, 90, 70, 100,
						80, 10, 80, 20, 80, 30, 80, 40, 80, 50, 80, 60, 80, 70, 80, 80, 80, 90, 80, 100,
						90, 10, 90, 20, 90, 30, 90, 40, 90, 50, 90, 60, 90, 70, 90, 80, 90, 90, 90, 100,
						100, 10, 100, 20, 100, 30, 100, 40, 100, 50, 100, 60, 100, 70, 100, 80, 100, 90, 100, 100}; // This variable contains the input data of the system under study for when n=100.
  double databaseY[] = {56.08, 47.12, 40.72, 36.88, 35.6, 36.88, 40.72, 47.12, 56.08, 67.6, 44.88, 35.92, 29.52, 25.68, 24.4, 25.68, 29.52, 35.92, 44.88, 56.4, 36.88, 27.92, 21.52, 17.68, 16.4, 17.68, 21.52, 27.92, 36.88, 48.4, 32.08, 23.12, 16.72, 12.88, 11.6, 12.88, 16.72, 23.12, 32.08, 43.6, 30.48, 21.52, 15.12, 11.28, 10, 11.28, 15.12, 21.52, 30.48, 42, 32.08, 23.12, 16.72, 12.88, 11.6, 12.88, 16.72, 23.12, 32.08, 43.6, 36.88, 27.92, 21.52, 17.68, 16.4, 17.68, 21.52, 27.92, 36.88, 48.4, 44.88, 35.92, 29.52, 25.68, 24.4, 25.68, 29.52, 35.92, 44.88, 56.4, 56.08, 47.12, 40.72, 36.88, 35.6, 36.88, 40.72, 47.12, 56.08, 67.6, 70.48, 61.52, 55.12, 51.28, 50, 51.28, 55.12, 61.52, 70.48, 82}; // This variable contains the expected/real output data of the system under study for when n=100.
  double b_ideal[5]; // This variable will be used to contain the ideal coefficient values that the model to be trained should give.
  b_ideal[0] = 82; // This is the true b_0 coefficient value with which the database was generated.
  b_ideal[1] = -1.6; // This is the true b_1 coefficient value with which the database was generated.
  b_ideal[2] = 0.016; // This is the true b_2 coefficient value with which the database was generated.
  b_ideal[3] = -1.28; // This is the true b_3 coefficient value with which the database was generated.
  b_ideal[4] = 0.0128; // This is the true b_4 coefficient value with which the database was generated.

  long int startingTime; // This variable will be used to store the starting timestamp in milliseconds of a certain process in which the time will be counted.
  long int elapsedTime; // This variable will be used to store the total time in milliseconds of a certain process in which the time will be counted.

  double differentiation; // Variable used to store the error obtained for a certain value.
  double epsilon; // Variable used to store the max error value permitted during validation process.
  char isMatch; // Variable used as a flag to indicate if the current comparison of values stands for a match. Note that the value of 1 = is a match and 0 = is not a match.
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
	  	printf("Initializing the output and input data with %d samples and %d independent variables...\n", n, m);
	  	startingTime = HAL_GetTick(); // We obtain the reference time to count the elapsed time to initialize the input data to be used.
	  	double *X = (double *) malloc(n*m*sizeof(double));
	  	double *Y = (double *) malloc(n*p*sizeof(double));
	  	// Create the output (Y) and input (X) data by using the same data as in the reference .csv file, but by duplicating the data until the desired number of samples is met.
	  	for (int currentIteration=0; currentIteration<(n/n_database); currentIteration++) {
	  		for (int currentRow=0; currentRow<n_database; currentRow++) {
	  			Y[currentRow + currentIteration*n_database] = databaseY[currentRow];
	  			// We apply the unrolling loop technique to store the data belonging to the current row of the input data.
	  			X[0 + currentRow*m + currentIteration*n_database*m] = databaseX[0 + currentRow*m];
	  			X[1 + currentRow*m + currentIteration*n_database*m] = databaseX[1 + currentRow*m];
	  		}
	  	}
	  	elapsedTime = HAL_GetTick() - startingTime; // We obtain the elapsed time to initialize the input data to be used.
	  	printf("Output and input data initialization elapsed %ld milliseconds.\n\n", elapsedTime);


	  	// ------------------------- DATA MODELING ----------------------- //
	  	printf("Initializing CenyML multiple polynomial regression algorithm ...\n");
	  	startingTime = HAL_GetTick(); // We obtain the reference time to count the elapsed time to apply the multiple polynomial regression with the input data (X).
	  	// Allocate the memory required for the variable "b", which will contain the identified best fitting coefficient values that will result from the multiple polynomial regression algorithm.
	  	double *b = (double *) calloc((m*N+1)*p, sizeof(double));
	  	getMultiplePolynomialRegression(X, Y, n, m, p, N, (char) 0, (char) 0, b); // NOTE: Remember that this functions stores the resulting coefficients in the pointer variable "b".
	  	elapsedTime = HAL_GetTick() - startingTime; // We obtain the elapsed time to apply the multiple polynomial regression with the input data (X).
	  	printf("CenyML multiple polynomial regression algorithm elapsed %ld milliseconds.\n\n", elapsedTime);


	  	// ------------------- VALIDATION OF THE MODEL ------------------- //
	  	// We validate the getMultiplePolynomialRegression method.
	  	printf("Initializing coefficients validation of the CenyML getMultiplePolynomialRegression method ...\n");
	  	startingTime = HAL_GetTick(); // We obtain the reference time to count the elapsed time to validate the getMultiplePolynomialRegression method.
	  	epsilon = 1.0E-8; // Variable used to store the max error value permitted during validation process.
	  	isMatch = 1; // Variable used as a flag to indicate if the current comparison of values stands for a match. Note that the value of 1 = is a match and 0 = is not a match.
	  	// We check that all the differentiations do not surpass the error indicated through the variable "epsilon".
	  	for (int currentRow=0; currentRow<m+1; currentRow++) {
			differentiation = fabs(b[currentRow] - b_ideal[currentRow]);
			if (differentiation > epsilon) { // if the error surpassed the value permitted, then terminate validation process and emit message to indicate a non match.
				isMatch = 0;
				printf("Validation process DID NOT MATCH! and a difference of %f was obtained.\n", differentiation);
				break;
			}
	  	}
	  	if (isMatch == 1) { // If the flag "isMatch" indicates a true/high value, then emit message to indicate that the validation process matched.
	  		printf("Validation process MATCHED!\n");
	  	}
	  	elapsedTime = HAL_GetTick() - startingTime; // We obtain the elapsed time to validate the getMultiplePolynomialRegression method.
	  	printf("The coefficients validation of the CenyML getMultiplePolynomialRegression method elapsed %ld milliseconds.\n\n", elapsedTime);
	  	printf("The program has been successfully completed!\n");


	  	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	  	free(X);
	  	free(Y);
	  	free(b);
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

