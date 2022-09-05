import numpy as np

class RegLogistica():

  
  #Declararemos el número de iteraciones y el learning rate
  def __init__(self, learning_rate, num_i):

    self.learning_rate = learning_rate
    self.num_i = num_i

  #Crearemos una función para entrenar el modelo
  def fit(self, X, Y):
    

    #Lo siguiente nos regresa la forma de la base de datos
    #m --> cuantas lineas de datos hay (filas) --> nos va a servir a la hora de derivar
    #n --> cuantos features tiene la base de datos (columnas) --> nos sirve para saber cuantos pesos necesitamos
    self.m, self.n = X.shape

    #Iniciamos los valores weight y bias

    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y

    #Usamos Gradient Descent para optimizar

    for i in range(num_i):
      self.weights_new()



  def weights_new(self):

    #Función Sigmoide  

    fun_s = 1 / (1 + np.exp(-(z)))

    #Es la fórmula --> w*X + b
    z = self.X.dot(self.w) + self.b

    #Derivadas (fórmulas)

    dw =(1/self.m)*np.dot(self.X.T, (fun_s - self.Y)) #Usamos T para poder multiplicar las matrices 

    db= (1/self.m)*np.sum(fun_s - self.Y)

    #Actualizamos weights y bias 

    self.w =self.w -self.learning_rate * dw

    self.b = self.b - self.learning_rate * db



  def pred(self):

    #La función sigmoide devuelve la probabilidad del valor que esta prediciendo
    #A nosotros nos interesa saber el valor como tal por lo que haremos lo siguiente

    Y_pred = 1 / (1 + np.exp(-(z))) 
    z = self.X.dot(self.w) + self.b

    Y_pred = np.where(Y_pred > 0.5,1,0)

    return Y_pred