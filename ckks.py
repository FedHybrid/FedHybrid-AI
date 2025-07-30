import numpy as np

# 정수 노이즈 샘플링 함수
def sample_integer_noise(size, max_noise=3):
    return np.random.randint(-max_noise, max_noise+1, size, dtype=np.int64)

# 모듈러스 X^4 + 1 적용 함수 (정수 기반)
def mod_x4_plus_1(poly):
    n = len(poly)
    if n <= 4:
        return poly
    result = np.zeros(4, dtype=np.int64)
    for i in range(n):
        idx = i % 4
        if (i // 4) % 2 == 1:
            result[idx] -= poly[i]
        else:
            result[idx] += poly[i]
    return result

# 정수 계수로 반올림 함수 (정수 기반)
def round_to_integer(coeffs):
    return np.array(
        [round(c) for c in coeffs],
        dtype=np.int64
    )

# 정수 기반 IDFT 구현
def ckks_idft(vec, N):
    M = 2 * N
    xi = np.exp(-2j * np.pi / M)
    roots = [xi**(2*k + 1) for k in range(N)]
    result = np.zeros(N, dtype=np.complex128)
    for j in range(N):
        for k in range(N):
            result[j] += vec[k] * np.conj(roots[k])**j
    result /= N
    return result

# 입력 슬롯 (정수 기반)
z1 = np.array([1, 3], dtype=np.int64)
z2 = np.array([2, 4], dtype=np.int64)

# 1) 평문 인코딩용 스케일: 10비트로 조정 (더 작은 스케일)
z_q = 1 << 10   # 2^10 = 1,024

# 2) 리스케일링용 스케일: 현재는 동일하게 설정
rescale_q = z_q


# 1) 정수 확장 (헤르미티안 대신 단순 복사)
z1_h = np.array([z1[0], z1[0], z1[1], z1[1]], dtype=np.int64)
z2_h = np.array([z2[0], z2[0], z2[1], z2[1]], dtype=np.int64)


# 2) 스케일링
w1 = z_q * z1_h
w2 = z_q * z2_h


# 3) IDFT → 정수 반올림 → 메시지 다항식 계수
m1 = round_to_integer(ckks_idft(w1, N=4))
m2 = round_to_integer(ckks_idft(w2, N=4))

# 4) 암호화 파라미터
s = np.array([1, 1, 0, 0], dtype=np.int64)
a = np.array([3, 2, 0, 0], dtype=np.int64)
e = sample_integer_noise(4, max_noise=3)  # 정수 노이즈 추가!


# 5) 스위칭키 생성
s2 = mod_x4_plus_1(np.convolve(s, s))
swk_a = a
swk_b = -mod_x4_plus_1(np.convolve(swk_a, s2)) + e
swk = (swk_a, swk_b)


# 6) 암호화 함수 (정수 기반)
def encrypt(m_coeffs):
    c0 = m_coeffs + mod_x4_plus_1(np.convolve(a, s)) + e
    c1 = -a
    return c0, c1

c0_1, c1_1 = encrypt(m1)
c0_2, c1_2 = encrypt(m2)


# 7) 곱셈
d0 = mod_x4_plus_1(np.convolve(c0_1, c0_2))
d1 = mod_x4_plus_1(np.convolve(c0_1, c1_2) + np.convolve(c1_1, c0_2))
d2 = mod_x4_plus_1(np.convolve(c1_1, c1_2))


# 8) 리스케일링
d0_ = d0 / rescale_q
d1_ = d1 / rescale_q
d2_ = d2 / rescale_q


# 9) 키 스위칭
e0 = mod_x4_plus_1(np.convolve(d2_, swk[0]))
e1 = mod_x4_plus_1(np.convolve(d2_, swk[1]))
c0_s = d0_ + e0
c1_s = d1_ + e1


# 10) 복호화 (Δ 나누지 않음) - 주석 처리
# dec_poly = c0_s + mod_x4_plus_1(np.convolve(c1_s, s))


# 11) 캐노니컬 임베딩 역변환 — 주석 처리
# N = 4
# M = 2 * N
# xi = np.exp(-2j * np.pi / M)
# roots = [xi**(2*k + 1) for k in range(N)]
# poly = np.poly1d(dec_poly[::-1])
# decoded = np.array([poly(r) for r in roots], dtype=np.complex128)

# 12) 슬롯 추출 & z_q 역스케일링 - 주석 처리
# recovered = decoded[[0, 2]] / z_q


# CKKS 암호문 덧셈 (리스케일링 없이)
def ckks_add(c1, c2):
    c0_sum = c1[0] + c2[0]
    c1_sum = c1[1] + c2[1]
    return c0_sum, c1_sum

# 예시: m1, m2 암호문 덧셈
c0_add, c1_add = ckks_add((c0_1, c1_1), (c0_2, c1_2))

# 복호화 - 주석 처리
# dec_poly_add = c0_add + mod_x4_plus_1(np.convolve(c1_add, s))
# poly_add = np.poly1d(dec_poly_add[::-1])
# decoded_add = np.array([poly_add(r) for r in roots], dtype=np.complex128)
# recovered_add = decoded_add[[0, 2]] / z_q


def ckks_minus(c1, c2):
    c0_sub = c1[0] - c2[0]
    c1_sub = c1[1] - c2[1]
    return c0_sub, c1_sub

def ckks_scale(c, scale_factor):
    """
    CKKS 암호문에 스케일 팩터를 곱하는 함수
    """
    c0_scaled = c[0] * scale_factor
    c1_scaled = c[1] * scale_factor
    return c0_scaled, c1_scaled

#예시 : m1, m2 암호문 뺄셈
c0_sub, c1_sub = ckks_minus((c0_1, c1_1), (c0_2, c1_2))

# 복호화 - 주석 처리
# dec_poly_sub = c0_sub + mod_x4_plus_1(np.convolve(c1_sub, s))
# poly_sub = np.poly1d(dec_poly_sub[::-1])
# decoded_sub = np.array([poly_sub(r) for r in roots], dtype=np.complex128)
# recovered_sub = decoded_sub[[0, 2]] / z_q


def decrypt(c0, c1, s, z_q, N):
    """
    CKKS 복호화: 
      1) 다항식 복호화 (레이어 1)
         dec_poly = c0(x) + mod_x4_plus_1(c1(x) * s(x))
      2) IDFT → 슬롯 복원
         m_vals = ckks_idft(dec_poly, N)
      3) z_q 역스케일링
         return m_vals / z_q
    """
    # 1) 다항식 복호화
    dec_poly = c0 + mod_x4_plus_1(np.convolve(c1, s))
    # 2) 슬롯 벡터 복원
    m_vals = ckks_idft(dec_poly, N)
    # 3) 스케일 역전
    return m_vals / z_q

def batch_encrypt(m_coeffs, batch_size=4):
    """
    큰 벡터를 배치로 나누어 정수 기반 암호화 수행
    """
    n = len(m_coeffs)
    num_batches = (n + batch_size - 1) // batch_size
    
    c0_list = []
    c1_list = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n)
        
        # 배치 크기에 맞게 패딩
        batch = np.zeros(batch_size, dtype=np.int64)
        batch[:end_idx-start_idx] = m_coeffs[start_idx:end_idx]
        
        # 정수 기반 암호화
        c0, c1 = encrypt(batch)
        c0_list.append(c0)
        c1_list.append(c1)
    
    return c0_list, c1_list

def batch_decrypt(c0_list, c1_list, original_size, batch_size=4):
    """
    배치로 암호화된 데이터를 복호화하여 원래 크기로 복원 (정수 기반)
    """
    decrypted_parts = []
    
    for c0, c1 in zip(c0_list, c1_list):
        # 정수 기반 복호화
        decrypted = decrypt(c0, c1, s, z_q, batch_size)
        decrypted_parts.append(decrypted)
    
    # 모든 배치를 연결
    full_decrypted = np.concatenate(decrypted_parts)
    
    # 원래 크기로 잘라내기
    return full_decrypted[:original_size]
