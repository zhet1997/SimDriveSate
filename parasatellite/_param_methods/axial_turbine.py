import numpy as np

__all__ = [
    'legacy_11_params'
]


def arc(xc, yc, r, alpha, beta, t):
    # improved version
    a = alpha - (alpha > beta) * np.pi * 2
    theta = (1 - t) * a + t * beta
    x = r * np.cos(theta) + xc
    y = r * np.sin(theta) + yc
    coords = np.concatenate(
        (np.expand_dims(x, -1), np.expand_dims(y, -1)), axis=1)
    return coords


def legacy_eparams_cp(
    cx, t, r_le, r_te, gamma,
    beta_in, wedge_is, wedge_ip,
    eff_beta_out, deflect_out, corr,
    wedge_out
):

    # Step 1: trailing edge center: [0, 0]
    # Step 2: leading edge center:
    p2_x = - (cx - r_le - r_te)
    p2_y = - p2_x * np.tan(gamma)
    LEC = np.array([p2_x, p2_y])

    # Step 3: LE arc endpoints determination
    l1_x = p2_x - r_le * np.cos(beta_in - wedge_is)
    l1_y = p2_y + r_le * np.sin(beta_in - wedge_is)
    P1 = np.array([l1_x, l1_y])
    K1 = 1 / np.tan(beta_in - wedge_is)

    l2_x = p2_x + r_le * np.cos(beta_in + wedge_ip)
    l2_y = p2_y - r_le * np.sin(beta_in + wedge_ip)
    P4 = np.array([l2_x, l2_y])
    K4 = np.tan(- beta_in - wedge_ip + np.pi / 2)

    # Step 4: duplicate trailing edge by grid distance
    p3_x = 0
    p3_y = t

    # Step 5: throat tangent circle
    rb = t * np.sin(eff_beta_out) + r_te

    # Step 6: common tangent
    an1 = np.arcsin((rb + r_te) / t)

    # Step 7: ss startpoint
    an2 = an1 - deflect_out
    t1_x = r_te * np.cos(an2)
    t1_y = r_te * np.sin(an2)
    P2 = np.array([t1_x, t1_y])
    K2 = -1 / np.tan(an2)

    # Step 8: intersection
    # 三正弦定理
    #
    # Debug Note:
    # $\dfrac{r_b}{\sin(an2)} = \dfrac{t - r_te / \sin(an2)}{\sin(\pi - an2 - an3)}$
    # 原实现中 $t - r_te / \sin(an2)$ 项未考虑后一部分
    #
    an3 = np.arcsin((t * np.sin(an2) - r_te) / rb) - an2
    g0_x = -rb * np.sin(an3)
    g0_y = t - rb * np.cos(an3)

    # Step 9: corr point
    g_x = g0_x * (1 - corr) + t1_x * corr
    g_y = g0_y * (1 - corr) + t1_y * corr

    # Step 10: tangent line from corr point with throat circle
    an4 = np.arccos(rb / np.sqrt((g_x - p3_x)**2 + (g_y - p3_y)**2))
    an5 = np.arctan((g_x - p3_x) / (g_y - p3_y))
    an6 = an4 + an5
    c3_x = - rb * np.sin(an6)
    c3_y = t - rb * np.cos(an6)
    P3 = np.array([c3_x, c3_y])
    K3 = - np.tan(an6)

    # Step 11: ps endpoint
    an7 = np.pi + an2 + wedge_out
    t2_x = r_te * np.cos(an7)
    t2_y = r_te * np.sin(an7)
    P5 = np.array([t2_x, t2_y])
    K5 = np.tan(an7 - np.pi / 2)

    return P1, P2, P3, P4, P5, K1, K2, K3, K4, K5, LEC, an2, an7


def legacy_11_params(cx, t, r_le, r_te, gamma, beta_in, wedge_is, wedge_ip, s_factor,
                   eff_beta_out, deflect_out, corr, wedge_out,
                   sl1, sl2, st1, st2, p1, p2):

    # Supersample
    n_sl = 40
    n_st = 40
    n_le_ss = 16
    n_le_ps = 16
    n_te_ss = 16
    n_te_ps = 16
    n_p = 80

    P1, P2, P3, P4, P5, K1, K2, K3, K4, K5, LEC, an2, an7 = legacy_eparams_cp(
        cx, t, r_le, r_te, gamma, beta_in, wedge_is, wedge_ip,
        eff_beta_out, deflect_out, corr, wedge_out)

    # Bezier sections

    MAT_B = np.array([[1, 0, 0, 0],
                      [-3, 3, 0, 0],
                      [3, -6, 3, 0],
                      [-1, 3, -3, 1]])

    MAT_B_O4 = np.array([[1, 0, 0, 0, 0],
                         [-4, 4, 0, 0, 0],
                         [6, -12, 6, 0, 0],
                         [-4, 12, -12, 4, 0],
                         [1, -4, 6, -4, 1]])

    sl_intersect_x = (K1 * P1[0] - K3 * P3[0] - P1[1] + P3[1]) / (K1 - K3)
    sl_intersect_y = K1 * (sl_intersect_x - P1[0]) + P1[1]

    st_intersect_x = (K3 * P3[0] - K2 * P2[0] - P3[1] + P2[1]) / (K3 - K2)
    st_intersect_y = K3 * (st_intersect_x - P3[0]) + P3[1]

    p_intersect_x = (K4 * P4[0] - K5 * P5[0] - P4[1] + P5[1]) / (K4 - K5)
    p_intersect_y = K4 * (p_intersect_x - P4[0]) + P4[1]

    # Increased-order implementation
    # ------------------------------------------------------------------------
    MAT_V_SL = np.zeros([5, 2])
    MAT_V_SL[0] = P3
    MAT_V_SL[4] = P1

    sl1_cp_x = P3[0] * (1 - sl2) + sl_intersect_x * sl2
    sl1_cp_y = P3[1] * (1 - sl2) + sl_intersect_y * sl2
    sl2_cp_x = P1[0] * (1 - sl1) + sl_intersect_x * sl1
    sl2_cp_y = P1[1] * (1 - sl1) + sl_intersect_y * sl1

    # s_factor activation
    MAT_V_SL[1, 0] = P3[0] * 0.25 + sl1_cp_x * 0.75
    MAT_V_SL[1, 1] = P3[1] * 0.25 + sl1_cp_y * 0.75
    MAT_V_SL[2, 0] = 0.5 * (sl1_cp_x + sl2_cp_x)
    MAT_V_SL[2, 1] = 0.5 * (sl1_cp_y + sl2_cp_y)
    MAT_V_SL[3, 0] = sl2_cp_x * s_factor + P1[0] * (1 - s_factor)
    MAT_V_SL[3, 1] = sl2_cp_y * s_factor + P1[1] * (1 - s_factor)

    u_sl = np.linspace(0, 1, n_sl + 2)[1:-1]
    MAT_U_SL = np.concatenate(([np.ones_like(u_sl)], [u_sl], [
                              u_sl**2], [u_sl**3], [u_sl**4]), axis=0)
    MAT_U_SL = np.transpose(MAT_U_SL)
    M1_SL = np.einsum('jk,kl->jl', MAT_U_SL, MAT_B_O4)
    X_SL = np.einsum('jk,kl->jl', M1_SL, MAT_V_SL)
    # -----------------------------------------------------------------------
    # 4th-order segment end

    MAT_V_ST = np.zeros([4, 2])
    MAT_V_ST[0] = P2
    MAT_V_ST[3] = P3
    MAT_V_ST[1, 0] = P2[0] * (1 - st2) + st_intersect_x * st2
    MAT_V_ST[1, 1] = P2[1] * (1 - st2) + st_intersect_y * st2
    MAT_V_ST[2, 0] = P3[0] * (1 - st1) + st_intersect_x * st1
    MAT_V_ST[2, 1] = P3[1] * (1 - st1) + st_intersect_y * st1
    u_st = np.linspace(0, 1, n_st + 1)[1:]
    MAT_U_ST = np.concatenate(
        ([np.ones_like(u_st)], [u_st], [u_st**2], [u_st**3]), axis=0)
    MAT_U_ST = np.transpose(MAT_U_ST)
    M1_ST = np.einsum('jk,kl->jl', MAT_U_ST, MAT_B)
    X_ST = np.einsum('jk,kl->jl', M1_ST, MAT_V_ST)

    MAT_V_P = np.zeros([4, 2])
    MAT_V_P[0] = P4
    MAT_V_P[3] = P5
    MAT_V_P[1, 0] = P4[0] * (1 - p1) + p_intersect_x * p1
    MAT_V_P[1, 1] = P4[1] * (1 - p1) + p_intersect_y * p1
    MAT_V_P[2, 0] = P5[0] * (1 - p2) + p_intersect_x * p2
    MAT_V_P[2, 1] = P5[1] * (1 - p2) + p_intersect_y * p2
    u_p = np.linspace(0, 1, n_p + 2)[1:-1]
    MAT_U_P = np.concatenate(
        ([np.ones_like(u_p)], [u_p], [u_p**2], [u_p**3]), axis=0)
    MAT_U_P = np.transpose(MAT_U_P)
    M1_P = np.einsum('jk,kl->jl', MAT_U_P, MAT_B)
    X_P = np.einsum('jk,kl->jl', M1_P, MAT_V_P)

    # Arc sections
    X_TE1 = arc(0, 0, r_te, (an2 + an7) / 2 - np.pi, an2, np.linspace(0, 1, n_te_ss))
    X_TE2 = arc(0, 0, r_te, an7, (an2 + an7) / 2 - np.pi, np.linspace(0, 1, n_te_ps))

    an8 = np.pi - beta_in + wedge_is
    an9 = 2 * np.pi - beta_in - wedge_ip
    X_LE1 = arc(LEC[0], LEC[1], r_le, an8, (an8 + an9) / 2, np.linspace(0, 1, n_le_ss))
    X_LE2 = arc(LEC[0], LEC[1], r_le, (an8 + an9) / 2, an9, np.linspace(0, 1, n_le_ps))

    X_raw_ss = np.concatenate((X_TE1, X_ST, X_SL, X_LE1), axis=0)
    X_raw_ss = np.flip(X_raw_ss, 0)
    X_raw_ps = np.concatenate((X_LE2, X_P, X_TE2), axis=0)
    # corr = np.stack((-r_te, np.zeros_like(r_te)), dim=1).unsqueeze(2).repeat(1, 1, X_raw.shape[2])
    # return X_raw + corr
    return X_raw_ss, X_raw_ps
