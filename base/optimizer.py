def gd(learning_rate, weight):
    return GradientDescent(learning_rate, weight)

'''
모멘텀
업데이트되는 값을 과거의 정보들을 활용하여 반영
vector값에 0~1인 작은 gamma값을 계속 누적 시킴으로서
누적 될 수록 예전 값은 버리고, 새로운 값에 가중치를 크게 두게된다.
'''
def Momentum(weight, learning_rate, gamma, last_vec):
    vec = (last_vec * gamma) + gd(learning_rate, weight)
    weight = weight - vec
    return weight, vec

'''
NAG
먼저 모멘텀을 활용하여 움직인 후
그 자리에서 그래디언트 값을 구한다.
'''
def Nesterov_Accelerated_Gradient(weight, learning_rate, gamma, last_vec):
    momentum = gamma * last_vec
    vec = momentum + gd(learning_rate, (weight - momentum))
    weight = weight - vec
    return weight, vec

'''
Adagrad
learning rate의 조절
Step size를 사용하기 시작
각각의 파라미터들 마다 업데이트를 다르게 한다.
가끔 업데이트되는 값에 대해서는 크게 업데이트를 하고
자주 업데이트되는 값에 대해서는 작게 업데이트를 한다.
sparse data에 대해서 suited
g값에서 제곱이 계속 누적되기 때문에 값이 커져서 나중에는 학습이 힘듦. 
'''
def Adagrad(weight, learning_rate, step_size, last_g, epsilon):
    g = last_g + gd(learning_rate, weight)**2
    weight = weight - step_size * gd(learning_rate, weight)**2 / (sqrt(g + epsilon))
    return weight, g

'''
RMSprop
Adagrad의 단점을 보완한 방법이다.
g값이 커지는걸 막기위해 지수평균을 도입 
'''
def RMSprop(weight, learning_rate, step_size, gamma, last_g, epsilon):
    g = (gamma * last_g) + (1 - gamma) * (gd(learning_rate, weight) ** 2)
    weight = weight - step_size * gd(learning_rate, weight) / (sqrt(g + epsilon))
    return weight, g

'''
AdaDelta
Adagrad 단점을 보완한 방법이다.
rms prop과 다르게 step size대신에 step size의 변화량을 넣어줬다. 
'''
def AdaDelta(weight, learning_rate, last_g, last_s, last_delta, gamma, epsilon):
    g = gamma * last_g + (1 - gamma) * (gd(learning_rate, weight) ** 2)
    s = gamma * last_s + (1 - gamma) * (last_delta ** 2)
    delta = sqrt(s + epsilon) / sqrt(g + epsilon) * gd(learning_rate, weight)
    weight = weight - delta
    return weight, g, s, delta

'''
Adam
RMSProp과 Momentum 방식을 합친 것 같은 방법이다.
'''
def Adam(weight, learning_rate, step_size, last_m, last_g, beta1, beta2, epsilon):
    m = beta1 * last_m + (1 - beta1) * gd(weight, learning_rate)
    g = beta2 * last_g + (1 - beta2) * gd(weight, learning_rate) ** 2
    beta1 *= beta1
    beta2 *= beta2
    m_hat = m / (1 - beta1)
    g_hat = g / (1 - beta2)
    weight = weight - step_size * m_hat / (sqrt(g_hat) + epsilon)
    return weight, m, g, beta1, beta2