#include "FiveWin.ch"

// Función principal para entrenar el Transformer
PROCEDURE Main()
   LOCAL d_model := 128
   LOCAL n_heads := 4
   LOCAL num_layers := 1
   LOCAL learning_rate := 0.001
   LOCAL max_epochs := 100
   LOCAL output, loss, d_output, epoch

   // Generar datos de entrada y salida aleatorios
   LOCAL src := hb_MatrixRandom(10, d_model)  // 10 secuencias de entrada
   LOCAL tgt := hb_MatrixRandom(10, d_model)  // 10 secuencias de objetivo

   // Crear instancia del Transformer
   LOCAL transformer := Transformer():New(num_layers, d_model, n_heads)

   FOR epoch := 1 TO max_epochs
      output := transformer:Forward(src, tgt)
      loss := hb_MatrixSum(hb_MatrixMultiply(hb_MatrixSubstract(output, tgt), ;
                        hb_MatrixTranspose(hb_MatrixSubstract(output, tgt)))) / (10 * d_model)

      // Gradiente de salida
      d_output := hb_MatrixSubstract(output, tgt)

      // Retropropagación y actualización de pesos
      transformer:Backward(d_output)
      ActualizarPesos(transformer, learning_rate)

      // Mostrar pérdida en cada época
      ? "Época:", epoch, "Pérdida:", loss
   NEXT
RETURN

// Clase para implementar MultiHeadAttention
CLASS MultiHeadAttention
   DATA d_model, n_heads
   DATA WQ, WK, WV, WO
   DATA dWQ, dWK, dWV, dWO
   // Cache para backpropagation
   DATA Q_proj
   DATA K_proj
   DATA V_proj
   DATA attention_scores
   DATA attention_probs   

   METHOD New(d_model, n_heads)
   METHOD Forward(Q, K, V)
   METHOD Backward(d_output) 
   METHOD InitGradients()   
ENDCLASS

METHOD New(d_model, n_heads) CLASS MultiHeadAttention
   LOCAL scale := 1.0 / Sqrt(d_model)
   
   // Guardamos d_model como propiedad de la clase
   ::d_model := d_model
   
   // Inicializar matrices de pesos
   ::WQ := hb_MatrixScale(hb_MatrixRandom(d_model, d_model), scale)
   ::WK := hb_MatrixScale(hb_MatrixRandom(d_model, d_model), scale)
   ::WV := hb_MatrixScale(hb_MatrixRandom(d_model, d_model), scale)
   ::WO := hb_MatrixScale(hb_MatrixRandom(d_model, d_model), scale)

   // Inicializar gradientes
   ::InitGradients() 

RETURN Self

METHOD InitGradients() CLASS MultiHeadAttention
   // Inicializar gradientes como matrices de ceros
   ::dWQ := hb_MatrixZero(::d_model, ::d_model)
   ::dWK := hb_MatrixZero(::d_model, ::d_model)
   ::dWV := hb_MatrixZero(::d_model, ::d_model)
   ::dWO := hb_MatrixZero(::d_model, ::d_model)
RETURN NIL

METHOD Forward(Q, K, V) CLASS MultiHeadAttention

   // Guardamos los valores proyectados para usar en backward
   ::Q_proj := hb_MatrixMultiply(Q, ::WQ)
   ::K_proj := hb_MatrixMultiply(K, ::WK)
   ::V_proj := hb_MatrixMultiply(V, ::WV)
   
   // Calcular scores de atención
   ::attention_scores := hb_MatrixDiv(hb_MatrixMultiply(::Q_proj, hb_MatrixTranspose(::K_proj)), ;
                                    Sqrt(Len(::Q_proj[1])))
   
   // Aplicar softmax para obtener probabilidades de atención
   ::attention_probs := Softmax(::attention_scores)
   
RETURN hb_MatrixMultiply(::attention_probs, ::V_proj)

METHOD Backward(d_output) CLASS MultiHeadAttention
   LOCAL attention_grad, dV, d_attention_scores, dQ, dK, dV

   // Reiniciar gradientes
   ::InitGradients()

   // Gradiente de la salida
   attention_grad := d_output

   // Gradiente para V y WO
   dV := hb_MatrixMultiply(hb_MatrixTranspose(::attention_probs), attention_grad)
   ::dWO := hb_MatrixMultiply(hb_MatrixTranspose(::V_proj), attention_grad)

   // Gradiente para los scores de atención (antes de softmax)
   d_attention_scores := hb_MatrixMultiply(attention_grad, hb_MatrixTranspose(::V_proj))
   d_attention_scores := hb_SoftmaxBackward(::attention_probs, d_attention_scores)

   // Gradiente para Q y K
   dQ := hb_MatrixMultiply(d_attention_scores, ::WK)
   dK := hb_MatrixMultiply(hb_MatrixTranspose(d_attention_scores), ::Q_proj)

   // Actualizar gradientes de los pesos
   ::dWQ := hb_MatrixMultiply(hb_MatrixTranspose(Q), dQ)
   ::dWK := hb_MatrixMultiply(hb_MatrixTranspose(K), dK)
   ::dWV := hb_MatrixMultiply(hb_MatrixTranspose(V), dV)

   // Devolver gradiente para el siguiente layer
RETURN dQ

// Clase para implementar el Transformer
CLASS Transformer
   VAR layers

   METHOD New(num_layers, d_model, n_heads)
   METHOD Forward(src, tgt)
   METHOD Backward(d_output)
ENDCLASS

METHOD New(num_layers, d_model, n_heads) CLASS Transformer
   local i
   ::layers := Array(num_layers)
   FOR i := 1 TO num_layers
      ::layers[i] := MultiHeadAttention():New(d_model, n_heads)
   NEXT
RETURN Self

METHOD Forward(src, tgt) CLASS Transformer
   LOCAL output := src, i
   FOR i := 1 TO Len(::layers)
      output := ::layers[i]:Forward(output, output, output)
   NEXT
RETURN output

METHOD Backward(d_output) CLASS Transformer
   local i
   FOR i := Len(::layers) TO 1 STEP -1
      ::layers[i]:Backward(d_output)
   NEXT
RETURN NIL

// Función para actualizar los pesos del Transformer
FUNCTION ActualizarPesos(transformer, learning_rate)
   local i, layer
   FOR i := 1 TO Len(transformer:layers)
      layer := transformer:layers[i]
      XBrowser( layer:dWQ )
      layer:WQ := hb_MatrixSubstract(layer:WQ, hb_MatrixScale(layer:dWQ, learning_rate))
      layer:WK := hb_MatrixSubstract(layer:WK, hb_MatrixScale(layer:dWK, learning_rate))
      layer:WV := hb_MatrixSubstract(layer:WV, hb_MatrixScale(layer:dWV, learning_rate))
      layer:WO := hb_MatrixSubstract(layer:WO, hb_MatrixScale(layer:dWO, learning_rate))
   NEXT
RETURN nil

FUNCTION hb_ArrayMax(aArray)
   LOCAL nMax := NIL
   LOCAL i

   // Verificar que el parámetro sea un array
   IF ValType(aArray) != "A"
      RETURN NIL
   ENDIF

   // Recorrer el array para encontrar el máximo
   FOR i := 1 TO Len(aArray)
      IF i == 1 .OR. aArray[i] > nMax
         nMax := aArray[i]
      ENDIF
   NEXT

RETURN nMax

#pragma BEGINDUMP

#include <hbapi.h>
#include <hbapiitm.h>
#include <hbapierr.h>
#include <math.h>

HB_FUNC( HB_MATRIXMULTIPLY )
{
   PHB_ITEM pMatrix1 = hb_param( 1, HB_IT_ARRAY ); // Primera matriz
   PHB_ITEM pMatrix2 = hb_param( 2, HB_IT_ARRAY ); // Segunda matriz

   if( pMatrix1 && pMatrix2 )
   {
      // Dimensiones de la primera matriz
      int rows1 = hb_arrayLen( pMatrix1 );
      PHB_ITEM pRow1, pRow2, pResult, pRowResult;
      int i, k, cols1, rows2, cols2;

      if( rows1 == 0 )
      {
         hb_errRT_BASE( EG_ARG, 3012, "First matrix is empty", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
         return;
      }
      pRow1 = hb_arrayGetItemPtr( pMatrix1, 1 );
      if( !pRow1 || !HB_IS_ARRAY( pRow1 ) )
      {
         hb_errRT_BASE( EG_ARG, 3012, "First matrix is not valid", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
         return;
      }
      cols1 = hb_arrayLen( pRow1 );

      // Dimensiones de la segunda matriz
      rows2 = hb_arrayLen( pMatrix2 );
      if( rows2 == 0 )
      {
         hb_errRT_BASE( EG_ARG, 3012, "Second matrix is empty", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
         return;
      }
      pRow2 = hb_arrayGetItemPtr( pMatrix2, 1 );
      if( !pRow2 || !HB_IS_ARRAY( pRow2 ) )
      {
         hb_errRT_BASE( EG_ARG, 3012, "Second matrix is not valid", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
         return;
      }
      cols2 = hb_arrayLen( pRow2 );

      // Validar compatibilidad para la multiplicación (cols1 debe ser igual a rows2)
      if( cols1 != rows2 )
      {
         hb_errRT_BASE( EG_ARG, 3012, "Matrix dimensions do not match for multiplication", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
         return;
      }

      // Crear la matriz de resultado (rows1 x cols2)
      pResult = hb_itemArrayNew( rows1 );

      for( i = 0; i < rows1; i++ )
      {
         PHB_ITEM pRowResult = hb_itemArrayNew( cols2 );
         hb_arraySet( pResult, i + 1, pRowResult );
         hb_itemRelease( pRowResult ); // Liberar referencia local
      }

      // Realizar la multiplicación de matrices
      for( i = 0; i < rows1; i++ )
      {
         PHB_ITEM pRowA = hb_arrayGetItemPtr( pMatrix1, i + 1 );
         int j;

         for( j = 0; j < cols2; j++ )
         {
            double sum = 0.0;
            for( k = 0; k < cols1; k++ )
            {
               double a = hb_arrayGetND( pRowA, k + 1 );
               PHB_ITEM pRowB = hb_arrayGetItemPtr( pMatrix2, k + 1 );
               double b = hb_arrayGetND( pRowB, j + 1 );
               sum += a * b;
            }
            
            pRowResult = hb_arrayGetItemPtr( pResult, i + 1 );
            hb_arraySetND( pRowResult, j + 1, sum );
         }
      }

      // Devolver la matriz de resultado
      hb_itemReturnRelease( pResult );
   }
   else
   {
      hb_errRT_BASE( EG_ARG, 3012, "Invalid parameters", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
   }
}

HB_FUNC( HB_MATRIXSCALE )
{
   PHB_ITEM pMatrix = hb_param( 1, HB_IT_ARRAY ); // Primer parámetro: matriz a escalar
   double scale = hb_parnd( 2 );                 // Segundo parámetro: escalar

   if( pMatrix )
   {
      HB_SIZE nRows = hb_arrayLen( pMatrix );
      HB_SIZE i, j;
      PHB_ITEM pMatrixResult = hb_itemArrayNew( nRows );

      // Copiar y escalar los datos
      for( i = 0; i < nRows; i++ )
      {
         PHB_ITEM pRow = hb_arrayGetItemPtr( pMatrix, i + 1 );
         HB_SIZE nCols = hb_arrayLen( pRow );

         PHB_ITEM pRowResult = hb_itemArrayNew( nCols );

         for( j = 0; j < nCols; j++ )
         {
            double value = hb_arrayGetND( pRow, j + 1 );
            hb_arraySetND( pRowResult, j + 1, value * scale );
         }

         hb_arraySet( pMatrixResult, i + 1, pRowResult );
         hb_itemRelease( pRowResult );
      }

      hb_itemReturnRelease( pMatrixResult );
   }
   else
   {
      hb_errRT_BASE( EG_ARG, 3012, NULL, HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
   }
}

HB_FUNC( HB_MATRIXDIV )
{
   PHB_ITEM pMatrix = hb_param( 1, HB_IT_ARRAY ); // Primer parámetro: matriz a escalar
   double scale = hb_parnd( 2 );                 // Segundo parámetro: escalar

   if( pMatrix )
   {
      HB_SIZE nRows = hb_arrayLen( pMatrix );
      HB_SIZE i, j;
      PHB_ITEM pMatrixResult = hb_itemArrayNew( nRows );

      // Copiar y escalar los datos
      for( i = 0; i < nRows; i++ )
      {
         PHB_ITEM pRow = hb_arrayGetItemPtr( pMatrix, i + 1 );
         HB_SIZE nCols = hb_arrayLen( pRow );

         PHB_ITEM pRowResult = hb_itemArrayNew( nCols );

         for( j = 0; j < nCols; j++ )
         {
            double value = hb_arrayGetND( pRow, j + 1 );
            hb_arraySetND( pRowResult, j + 1, value / scale );
         }

         hb_arraySet( pMatrixResult, i + 1, pRowResult );
         hb_itemRelease( pRowResult );
      }

      hb_itemReturnRelease( pMatrixResult );
   }
   else
   {
      hb_errRT_BASE( EG_ARG, 3012, NULL, HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
   }
}

HB_FUNC( HB_MATRIXTRANSPOSE )
{
   PHB_ITEM pMatrix = hb_param( 1, HB_IT_ARRAY ); // Primer parámetro: matriz a transponer

   if( pMatrix )
   {
      HB_SIZE nRows = hb_arrayLen( pMatrix );
      HB_SIZE nCols = hb_arrayLen( hb_arrayGetItemPtr( pMatrix, 1 ) ); // Número de columnas de la primera fila
      HB_SIZE i, j;

      PHB_ITEM pMatrixResult = hb_itemArrayNew( nCols ); // Crear matriz transpuesta (nCols x nRows)

      // Inicializar las filas de la matriz transpuesta
      for( i = 0; i < nCols; i++ )
      {
         hb_arraySet( pMatrixResult, i + 1, hb_itemArrayNew( nRows ) );
      }

      // Rellenar la matriz transpuesta
      for( i = 0; i < nRows; i++ )
      {
         PHB_ITEM pRow = hb_arrayGetItemPtr( pMatrix, i + 1 );
         for( j = 0; j < nCols; j++ )
         {
            double value = hb_arrayGetND( pRow, j + 1 ); // Obtener el valor original
            PHB_ITEM pTransposedRow = hb_arrayGetItemPtr( pMatrixResult, j + 1 );
            hb_arraySetND( pTransposedRow, i + 1, value ); // Asignar a la posición transpuesta
         }
      }

      hb_itemReturnRelease( pMatrixResult ); // Devolver la matriz transpuesta
   }
   else
   {
      hb_errRT_BASE( EG_ARG, 3012, NULL, HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
   }
}

HB_FUNC( HB_MATRIXZERO )
{
   HB_SIZE nRows = hb_parns( 1 ); // Número de filas
   HB_SIZE nCols = hb_parns( 2 ); // Número de columnas

   if( nRows > 0 && nCols > 0 )
   {
      HB_SIZE i, j;

      PHB_ITEM pMatrix = hb_itemArrayNew( nRows ); // Crear la matriz de nRows filas

      // Inicializar la matriz con ceros
      for( i = 0; i < nRows; i++ )
      {
         PHB_ITEM pRow = hb_itemArrayNew( nCols ); // Crear una fila con nCols columnas
         for( j = 0; j < nCols; j++ )
         {
            hb_arraySetND( pRow, j + 1, 0.0 ); // Establecer cada elemento a 0.0
         }
         hb_arraySet( pMatrix, i + 1, pRow ); // Añadir la fila a la matriz
         hb_itemRelease( pRow ); // Liberar la fila temporal
      }

      hb_itemReturnRelease( pMatrix ); // Devolver la matriz completa
   }
   else
   {
      hb_errRT_BASE( EG_ARG, 3012, NULL, HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
   }
}

HB_FUNC( HB_MATRIXRANDOM )
{
   HB_SIZE nRows = hb_parns( 1 ); // Número de filas
   HB_SIZE nCols = hb_parns( 2 ); // Número de columnas

   if( nRows > 0 && nCols > 0 )
   {
      HB_SIZE i, j;

      PHB_ITEM pMatrix = hb_itemArrayNew( nRows ); // Crear la matriz de nRows filas

      // Inicializar la matriz con valores aleatorios
      for( i = 0; i < nRows; i++ )
      {
         PHB_ITEM pRow = hb_itemArrayNew( nCols ); // Crear una fila con nCols columnas
         for( j = 0; j < nCols; j++ )
         {
            double randomValue = (double)rand() / RAND_MAX; // Valor aleatorio entre 0.0 y 1.0
            hb_arraySetND( pRow, j + 1, randomValue );
         }
         hb_arraySet( pMatrix, i + 1, pRow ); // Añadir la fila a la matriz
         hb_itemRelease( pRow ); // Liberar la fila temporal
      }

      hb_itemReturnRelease( pMatrix ); // Devolver la matriz completa
   }
   else
   {
      hb_errRT_BASE( EG_ARG, 3012, NULL, HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
   }
}

HB_FUNC( HB_SOFTMAX )
{
   PHB_ITEM pValues = hb_param( 1, HB_IT_ARRAY ); // Primer parámetro: array multidimensional de valores

   if( pValues )
   {
      int nRows = hb_arrayLen( pValues ); // Número de filas
      if( nRows > 0 )
      {
         // Asumimos que las filas tienen la misma longitud
         PHB_ITEM pFirstRow = hb_arrayGetItemPtr( pValues, 1 );
         int nCols = hb_arrayLen( pFirstRow ); // Número de columnas (basado en la primera fila)

         PHB_ITEM pResult = hb_itemArrayNew( nRows ); // Array para almacenar los resultados
         int i, j;

         // Recorrer cada fila
         for( i = 0; i < nRows; i++ )
         {
            PHB_ITEM pRow = hb_arrayGetItemPtr( pValues, i + 1 );
            PHB_ITEM pRowResult = hb_itemArrayNew( nCols ); // Fila de resultados para Softmax

            double* expValues = (double*) hb_xgrab( nCols * sizeof(double) );
            double sumExp = 0.0;

            // Calcular e^x para cada elemento de la fila y la suma total
            for( j = 0; j < nCols; j++ )
            {
               double value = hb_arrayGetND( pRow, j + 1 );
               expValues[j] = pow( M_E, value );
               sumExp += expValues[j];
            }

            // Calcular Softmax para la fila dividiendo cada e^x por la suma total
            for( j = 0; j < nCols; j++ )
            {
               double softmaxValue = expValues[j] / sumExp;
               hb_arraySetND( pRowResult, j + 1, softmaxValue );
            }

            hb_xfree( expValues ); // Liberar memoria para los exponentes

            // Guardar la fila de resultados en la matriz resultante
            hb_arraySet( pResult, i + 1, pRowResult );
            hb_itemRelease( pRowResult ); // Liberar la fila de resultados
         }

         hb_itemReturnRelease( pResult ); // Devolver la matriz de resultados
      }
      else
      {
         hb_errRT_BASE( EG_ARG, 3012, NULL, HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
      }
   }
   else
   {
      hb_errRT_BASE( EG_ARG, 3012, NULL, HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
   }
}

HB_FUNC( HB_SOFTMAXBACKWARD )
{
   PHB_ITEM pProbs = hb_param(1, HB_IT_ARRAY); // Softmax probabilities
   PHB_ITEM pGrad = hb_param(2, HB_IT_ARRAY);  // Upstream gradient

   if (pProbs && pGrad)
   {
      HB_SIZE nRows = hb_arrayLen(pProbs), nCols;
      PHB_ITEM pFirstRow, pResult, i, j, k;

      if (nRows == 0 || hb_arrayLen(pGrad) != nRows)
      {
         hb_errRT_BASE(EG_ARG, 3012, "Invalid matrix dimensions", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS);
         return;
      }

      pFirstRow = hb_arrayGetItemPtr(pProbs, 1);
      nCols = hb_arrayLen(pFirstRow);
      if (nCols == 0 || hb_arrayLen(hb_arrayGetItemPtr(pGrad, 1)) != nCols)
      {
         hb_errRT_BASE(EG_ARG, 3012, "Column dimensions do not match", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS);
         return;
      }

      // Create result matrix (nRows x nCols)
      pResult = hb_itemArrayNew(nRows);

      // Process each row
      for (i = 0; i < nRows; i++)
      {
         PHB_ITEM pProbRow = hb_arrayGetItemPtr(pProbs, i + 1);
         PHB_ITEM pGradRow = hb_arrayGetItemPtr(pGrad, i + 1);
         PHB_ITEM pResultRow = hb_itemArrayNew(nCols);
         
         // Compute gradient for each element in the row
         for (j = 0; j < nCols; j++)
         {
            double sum = 0.0;
            double prob_j = hb_arrayGetND(pProbRow, j + 1);
            double grad_j = hb_arrayGetND(pGradRow, j + 1);

            for (k = 0; k < nCols; k++)
            {
               double prob_k = hb_arrayGetND(pProbRow, k + 1);
               double grad_k = hb_arrayGetND(pGradRow, k + 1);
               if (j == k)
               {
                  sum += prob_j * (1.0 - prob_j) * grad_k;
               }
               else
               {
                  sum += -prob_j * prob_k * grad_k;
               }
            }
            hb_arraySetND(pResultRow, j + 1, sum);
         }

         hb_arraySet(pResult, i + 1, pResultRow);
         hb_itemRelease(pResultRow);
      }

      hb_itemReturnRelease(pResult);
   }
   else
   {
      hb_errRT_BASE(EG_ARG, 3012, "Invalid parameters", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS);
   }
}

HB_FUNC( HB_MATRIXSUBSTRACT )
{
   PHB_ITEM pMatrix1 = hb_param( 1, HB_IT_ARRAY ); // Primera matriz
   PHB_ITEM pMatrix2 = hb_param( 2, HB_IT_ARRAY ); // Segunda matriz

   if( pMatrix1 && pMatrix2 )
   {
      HB_SIZE nRows1 = hb_arrayLen( pMatrix1 );
      HB_SIZE nRows2 = hb_arrayLen( pMatrix2 );

      if( nRows1 == nRows2 && nRows1 > 0 )
      {
         HB_SIZE nCols1 = hb_arrayLen( hb_arrayGetItemPtr( pMatrix1, 1 ) );
         HB_SIZE nCols2 = hb_arrayLen( hb_arrayGetItemPtr( pMatrix2, 1 ) );

         if( nCols1 == nCols2 && nCols1 > 0 )
         {
            HB_SIZE i, j;

            // Crear la matriz de resultado
            PHB_ITEM pMatrixResult = hb_itemArrayNew( nRows1 );

            // Realizar la resta elemento a elemento
            for( i = 0; i < nRows1; i++ )
            {
               PHB_ITEM pRow1 = hb_arrayGetItemPtr( pMatrix1, i + 1 );
               PHB_ITEM pRow2 = hb_arrayGetItemPtr( pMatrix2, i + 1 );

               PHB_ITEM pRowResult = hb_itemArrayNew( nCols1 );

               for( j = 0; j < nCols1; j++ )
               {
                  double value1 = hb_arrayGetND( pRow1, j + 1 );
                  double value2 = hb_arrayGetND( pRow2, j + 1 );
                  hb_arraySetND( pRowResult, j + 1, value1 - value2 ); // Resta
               }

               hb_arraySet( pMatrixResult, i + 1, pRowResult ); // Añadir la fila al resultado
               hb_itemRelease( pRowResult ); // Liberar la fila temporal
            }

            hb_itemReturnRelease( pMatrixResult ); // Devolver la matriz resultado
         }
         else
         {
            // Error: Las columnas no coinciden
            hb_errRT_BASE( EG_ARG, 3012, "Column dimensions do not match", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
         }
      }
      else
      {
         // Error: Las filas no coinciden
         hb_errRT_BASE( EG_ARG, 3012, "Row dimensions do not match", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
      }
   }
   else
   {
      // Error: Argumentos inválidos
      hb_errRT_BASE( EG_ARG, 3012, "Invalid parameters", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
   }
}

HB_FUNC( HB_MATRIXSUM )
{
   PHB_ITEM pMatrix = hb_param( 1, HB_IT_ARRAY ); // Matriz de entrada

   if( pMatrix )
   {
      int nRows = hb_arrayLen( pMatrix ); // Número de filas

      if( nRows > 0 )
      {
         double sum = 0.0;
         int i;

         for( i = 0; i < nRows; i++ )
         {
            PHB_ITEM pRow = hb_arrayGetItemPtr( pMatrix, i + 1 );
            int nCols = hb_arrayLen( pRow ); // Número de columnas
            int j;

            for( j = 0; j < nCols; j++ )
            {
               sum += hb_arrayGetND( pRow, j + 1 ); // Sumar el elemento actual
            }
         }

         hb_retnd( sum ); // Devolver la suma como resultado
      }
      else
      {
         // Error: Matriz vacía
         hb_errRT_BASE( EG_ARG, 3012, "Empty matrix", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
      }
   }
   else
   {
      // Error: Argumentos inválidos
      hb_errRT_BASE( EG_ARG, 3012, "Invalid parameter", HB_ERR_FUNCNAME, HB_ERR_ARGS_BASEPARAMS );
   }
}

#pragma ENDDUMP
