import tensorflow as tf

# anterior perdida con tamaño variable
class VlaLoss(tf.keras.losses.Loss):
    def __init__(self, vocab_size, n_gram=4, bits_per_token=11, **kwargs):
        """
        vocab_size: tamaño del vocabulario
        n_gram: número de tokens por n-grama
        bits_per_token: longitud del código Gray de cada token
        """
        super().__init__(**kwargs)
        self.vocab_size = tf.constant(vocab_size, dtype=tf.float32)
        self.n_gram = n_gram
        self.bits_per_token = bits_per_token

    def gray_to_binary(self, gray_code):
        gray_bits = tf.cast(gray_code, tf.float32)

        def xor(prev, curr):
            return prev + curr - 2 * prev * curr

        first_bit = gray_bits[:, 0:1]
        rest_bits = gray_bits[:, 1:]
        rest_bits_T = tf.transpose(rest_bits, perm=[1, 0])
        binary_rest = tf.scan(xor, rest_bits_T, initializer=first_bit[:, 0])
        binary_rest = tf.transpose(binary_rest, perm=[1, 0])
        binary = tf.concat([first_bit, binary_rest], axis=1)
        return binary

    def binary_decimal(self, binary_pos):
        binary_pos = tf.cast(binary_pos, tf.float32)
        num_bits = tf.shape(binary_pos)[1]
        powers = tf.pow(2.0, tf.cast(tf.range(num_bits - 1, -1, -1), tf.float32))
        decimal_value = tf.reduce_sum(binary_pos * powers, axis=1)
        return decimal_value / self.vocab_size

    def call(self, y_true, y_pred):
        # Asegurar que los tensores sean float
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Dividir en subcódigos por token
        bits = self.bits_per_token
        n = self.n_gram
        total_bits = bits * n

        # (batch, total_bits) → (batch, n, bits)
        y_true_parts = tf.reshape(y_true, (-1, n, bits))
        y_pred_parts = tf.reshape(y_pred, (-1, n, bits))

        total_true = []
        total_pred = []

        for i in range(n):
            true_part = y_true_parts[:, i, :]
            pred_part = y_pred_parts[:, i, :]

            true_bin = self.gray_to_binary(true_part)
            pred_bin = self.gray_to_binary(pred_part)

            true_dec = self.binary_decimal(true_bin)
            pred_dec = self.binary_decimal(pred_bin)

            total_true.append(true_dec)
            total_pred.append(pred_dec)

        # Sumar los decimales de cada subcadena
        y_true_decimal = tf.add_n(total_true)
        y_pred_decimal = tf.add_n(total_pred)
    
        # y_true_decimal = tf.add_n(total_true) / self.n_gram
        # y_pred_decimal = tf.add_n(total_pred) / self.n_gram

        # Calcular diferencia logarítmica base 2
        diff = tf.abs(y_true_decimal - y_pred_decimal)
        log_diff = tf.math.log(diff + 1) / tf.math.log(2.0)

        return tf.reduce_mean(log_diff)

# anterior pérdida con n gramas fijos de 1
class FlaLoss(tf.keras.losses.Loss):
  def __init__(self, vocab_size, **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = tf.constant(vocab_size, dtype=tf.float32)


  def gray_to_binary(self,gray_code):
    #asegura tipo float32
    gray_bits = tf.cast(gray_code, tf.float32)

    #obtener el número de bits
    # num_bits = tf.shape(gray_code)[1]
    # Función para calcular XOR
    def xor(prev, curr):
      return prev + curr - 2 * prev * curr

    #tf scan sobre eje de bits 
    first_bit = gray_bits[:, 0:1]  # shape (batch,1)
    rest_bits = gray_bits[:, 1:]    # shape (batch, num_bits-1)
    
    rest_bits_T = tf.transpose(rest_bits, perm=[1,0])
    binary_rest = tf.scan(xor, rest_bits_T, initializer=first_bit[:,0])
    binary_rest = tf.transpose(binary_rest, perm=[1,0])
    binary = tf.concat([first_bit, binary_rest], axis=1)
    return binary 



  def binary_decimal(self, binary_pos):
    # Asegurar tipo float32 para operaciones
    binary_pos = tf.cast(binary_pos, tf.float32)
    #obtiene elnúmero de bits
    num_bits = tf.shape(binary_pos)[1]

    #crea pesos para cada bit  2^(bit-1-i)
    powers = tf.pow(2.0, tf.cast(tf.range(num_bits - 1, -1, -1), tf.float32))

    # Multiplicar cada bit por su peso y sumar por fila (batch)
    decimal_value = tf.reduce_sum(binary_pos * powers, axis=1)

    return decimal_value/self.vocab_size

  def call(self, y_true, y_pred):
    y_true_bin=self.gray_to_binary(y_true)
    y_pred_bin=self.gray_to_binary(y_pred)

    #binario poscicional 
    y_true_decimal=self.binary_decimal(y_true_bin)
    y_pred_decimal=self.binary_decimal(y_pred_bin)
    # print(y_true_decimal)
    # print(y_pred_decimal)

    #calculo de diferencia absoluta logaritmica:
    diff = tf.abs(y_true_decimal - y_pred_decimal)
    log_diff = tf.math.log(diff + 1) / tf.math.log(2.0)
    return tf.reduce_mean(log_diff)


#new with cosine similarity
class LacsLoss(tf.keras.losses.Loss):
  def __init__(self, vocab_size, bits_p_word, total_bits, alpha=0.7, **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = tf.constant(vocab_size, dtype=tf.float32)
    self.total_bits = int(total_bits)
    self.bits_p_word = int(bits_p_word)
    self.alpha=alpha

  def gray_to_binary(self,gray_code):
    #asegura tipo float32
    gray_bits = tf.cast(gray_code, tf.float32)

    #obtener el número de bits
    # num_bits = tf.shape(gray_code)[1]
    # Función para calcular XOR
    def xor(prev, curr):
      return prev + curr - 2 * prev * curr

    #tf scan sobre eje de bits
    first_bit = gray_bits[:, 0:1]  # shape (batch,1)
    rest_bits = gray_bits[:, 1:]    # shape (batch, num_bits-1)

    rest_bits_T = tf.transpose(rest_bits, perm=[1,0])
    binary_rest = tf.scan(xor, rest_bits_T, initializer=first_bit[:,0])
    binary_rest = tf.transpose(binary_rest, perm=[1,0])
    binary = tf.concat([first_bit, binary_rest], axis=1)
    return binary



  def binary_decimal(self, binary_pos):
    # Asegurar tipo float32 para operaciones
    binary_pos = tf.cast(binary_pos, tf.float32)
    #obtiene elnúmero de bits
    num_bits = tf.shape(binary_pos)[1]

    #crea pesos para cada bit  2^(bit-1-i)
    powers = tf.pow(2.0, tf.cast(tf.range(num_bits - 1, -1, -1), tf.float32))

    # Multiplicar cada bit por su peso y sumar por fila (batch)
    decimal_value = tf.reduce_sum(binary_pos * powers, axis=1)

    return decimal_value/self.vocab_size

  def words_per_sentence(self,chain_bin):

    return [chain_bin[:, i:i+self.bits_p_word] for i in range(0, self.total_bits, self.bits_p_word)]

  def compute_diferences(self, y_true_sections, y_pred_sections):

    losses=[]
    for y_t, y_p in zip(y_true_sections, y_pred_sections):
      y_true_bin=self.gray_to_binary(y_t)
      y_pred_bin=self.gray_to_binary(y_p)

      y_true_dec=self.binary_decimal(y_true_bin)
      y_pred_dec=self.binary_decimal(y_pred_bin)

      dif=tf.abs(y_true_dec-y_pred_dec)
      log_dif=tf.math.log(dif+1)/tf.math.log(2.0)
      losses.append(tf.reduce_mean(log_dif))
    return tf.reduce_mean(losses)

  def contrast_loss(self, y_true, y_pred):
    y_true_norm=tf.nn.l2_normalize(y_true, axis=1)
    y_pred_norm=tf.nn.l2_normalize(y_pred, axis=1)
    
    cos_sim=tf.reduce_sum(y_true_norm*y_pred_norm, axis=1)
    return tf.reduce_mean(1-cos_sim)





  def call(self, y_true, y_pred):
    # y_true_bin=self.gray_to_binary(y_true)
    # y_pred_bin=self.gray_to_binary(y_pred)

    y_true_sections=self.words_per_sentence(y_true)
    y_pred_sections=self.words_per_sentence(y_pred)
    print(y_true_sections)
    print(y_pred_sections)

    # print(self.bits_p_word)
    # print(self.total_bits)

    #binario poscicional
    # y_true_decimal=self.binary_decimal(y_true_bin)
    # y_pred_decimal=self.binary_decimal(y_pred_bin)
    # # print(y_true_decimal)
    # print(y_pred_decimal)

    #calculo de diferencia absoluta logaritmica:
    # diff = tf.abs(y_true_decimal - y_pred_decimal)
    # log_diff = tf.math.log(diff + 1) / tf.math.log(2.0)
    # return tf.reduce_mean(log_diff)

    reconstructive_loss=self.compute_diferences(y_true_sections, y_pred_sections)
    contrastive_loss=self.contrast_loss(y_true, y_pred)

    return self.alpha*reconstructive_loss+(1-self.alpha)*contrastive_loss
