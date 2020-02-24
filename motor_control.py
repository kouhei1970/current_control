#!/usr/bin/env python
# coding: utf-8

# #### 必要なモジュールのインポート

# In[ ]:

import numpy as np

from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
import ipywidgets as widgets
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Span, CrosshairTool, HoverTool, ResetTool, PanTool, WheelZoomTool


output_notebook()


# #### 自作制御用ライブラリ

# ##### 伝達関数の定義
# tf(num,den)
# - num:分子の係数リスト
# - den:分母の係数リスト

# In[ ]:


def tf(num,den):
    
    num=np.poly1d(num)
    den=np.poly1d(den)
    
    z=num.r
    p=den.r

    
    zreal=z[np.isclose(z.imag,0.0)].real
    zcomp=z[~np.isclose(z.imag,0.0)]
    preal=p[np.isclose(p.imag,0.0)].real
    pcomp=p[~np.isclose(p.imag,0.0)]

    zzreal=zreal
    zzcomp=zcomp
    ppreal=preal
    ppcomp=pcomp
    
    #分母分子から共通の極を見つけ出して削除する
    for x in zreal:
        ppreal=ppreal[~np.isclose(ppreal, x)]

    for x in preal:
        zzreal=zzreal[~np.isclose(zzreal, x)]

    for x in zcomp:
        ppcomp=ppcomp[~np.isclose(ppcomp, x)]

    for x in pcomp:
        zzcomp=zzcomp[~np.isclose(zzcomp, x)]
    
    zz=np.concatenate([zzreal, zzcomp], 0)
    pp=np.concatenate([ppreal, ppcomp], 0)
    
    
    num=np.poly1d(zz, r=True, variable = "s")*num.coef[0]
    den=np.poly1d(pp, r=True, variable = "s")*den.coef[0]

    return [num, den]


# ##### 伝達関数演算ライブラリ
# 
# - tf_add(sys1, sys2):伝達関数同士の和
# - tf_multi(sys1, sys2):伝達関数同士の積
# - tf_inv(sys):伝達関数の逆
#  - sys1,sys2,sys:伝達関数 

# In[ ]:


def tf_add(sys1, sys2):
    num=sys1[0]*sys2[1]+sys1[1]*sys2[0]
    den=sys1[1]*sys2[1]
    #print('d',num)
    #print('d',den)
    return  tf(num.coef, den.coef)


def tf_multi(sys1, sys2):
    num=sys1[0]*sys2[0]
    den=sys1[1]*sys2[1]
    return tf(num.coef, den.coef)

def tf_inv(sys):
    return tf(sys[1].coef, sys[0].coef)


# ##### ステップ応答とランプ応答
# - step(sys,st,et,step,debug)
# - ramp(sys,st,et,step,debug)
#  - sys:伝達関数
#  - st:初期時間
#  - et:最終時間
#  - step:計算ポイント数(デフォルト1000)
#  - debug:デバグスイッチ（デフォルトFalse）
# 

# In[ ]:


def step(sys,st,et,step=1000, debug=False):

    n=len(sys[1])
    p=sys[1].r
    if debug==True:
        print('order={}'.format(n))
        print('Pole={}'.format(p))
    
    y=np.zeros(step)
    t=np.linspace(st, et, step)

    for i in range(n):
        k=sys[0](p[i])/sys[1].deriv()(p[i])/p[i]
        
        if debug==True:
            print('k{}={}'.format(i+1,k))

        y=y+(k*np.exp(p[i]*t)).real
    
    k=sys[0](0)/sys[1](0)
    if debug==True:
        print('k{}={}'.format(i+2,k))

    y=y+k
    
    return t,y    


# In[ ]:


def ramp(sys, st, et, alpha=1.0, step=1000, debug=False):

    n=len(sys[1])
    p=sys[1].r
    if debug==True:
        print('order={}'.format(n))
        print('Pole={}'.format(p))
    
    y=np.zeros(step)
    t=np.linspace(st, et, step)

    for i in range(n):
        k=alpha*sys[0](p[i])/sys[1].deriv()(p[i])/p[i]/p[i]
        
        if debug==True:
            print('k{}={}'.format(i+1,k))

        y=y+(k*np.exp(p[i]*t)).real
    
    k=alpha*sys[0](0)/sys[1](0)
    if debug==True:
        print('k{}={}'.format(i+2,k))

    y=y+k*t
    
    k=alpha*(sys[0].deriv()(0)*sys[1](0)-sys[0](0)*sys[1].deriv()(0))/(sys[1](0))**2
    if debug==True:
        print('k{}={}'.format(i+3,k))
    
    y=y+k
    
    return t,y    


# ##### 出力シフト関数

# In[ ]:


#出力を遅らせる関数
def shift(x, s):
    l=len(x)
    x=np.roll(x, s)
    dummy_ones =np.ones(l-s)
    dummy_zeros=np.zeros(s)
    dummy=np.concatenate([dummy_zeros,dummy_ones],0)
    return x*dummy


# In[ ]:





# In[ ]:


def motor1724():
    #1724DCモータの諸元
    R=3.41
    K=6.59e-3
    L=75e-6
    D=1.4e-7
    J=1e-7

    return tf([0,K],[J*L, D*L+J*R, D*R+K**2])


# #### 台形制御計算関数

# In[ ]:


def Trapezoid_control(sys, cont, steps=100000):

    tf_one=tf(1,1)
    sys_open=tf_multi(cont, sys)
    sys_den=tf_add( tf_one, sys_open )
    sys_y=tf_multi( sys_open, tf_inv(sys_den))
    sys_u=tf_multi(cont,  tf_inv(sys_den))
    sys_e=tf_multi(tf_one,  tf_inv(sys_den))

    starttime=0.0
    endtime=1.0
    maxvalue=1000
    
    t,y1=ramp(sys_y, starttime, endtime, step=steps)
    t,u1=ramp(sys_u, starttime, endtime, step=steps)
    t,e1=ramp(sys_e, starttime, endtime, step=steps)

    #出力を遅らせて合成
    y2=shift(y1,int(steps/4))
    y3=shift(y1,int(2*steps/4))
    y4=shift(y1,int(3*steps/4))
    y=y1-y2-y3+y4
    y=maxvalue*y

    u2=shift(u1,int(steps/4))
    u3=shift(u1,int(2*steps/4))
    u4=shift(u1,int(3*steps/4))
    u=u1-u2-u3+u4
    u=maxvalue*u

    e2=shift(e1,int(steps/4))
    e3=shift(e1,int(2*steps/4))
    e4=shift(e1,int(3*steps/4))
    e=e1-e2-e3+e4
    e=maxvalue*e    

    return t, y, u, e


# ##### Bokehを用いた、インタラクティブなグラフ表示

# In[ ]:


def draw_control_result_chart(t,y,u,e,yr):
    p1 = figure(plot_height=200, 
                plot_width=800,
                background_fill_color='#efefef',
                y_axis_label='Omega(rad/s)'
    )
    p1.line(t, y, color="#ff0000", line_width=1.5, alpha=0.8)

    p2 = figure(plot_height=200,
                plot_width=800,
                background_fill_color='#efefef',
                y_axis_label='Input(V)'
    )
    p2.line(t, u, color="#008800", line_width=1.5, alpha=0.8)

    p3 = figure(plot_height=230,
                plot_width=800,
                background_fill_color='#efefef',
                y_range=[-yr, yr],
                x_axis_label='Time(s)',
                y_axis_label='Error(rad/s)'
    )
    p3.line(t, e, color="#000088", line_width=1.5, alpha=0.8)

    #show(p1)
    show(column(p1, p2, p3))    
    


# In[ ]:


def demo_sim(kp=0.1, ki=10, yr=0.8):
    
    motor=motor1724()
    cont=tf([kp, ki],[1, 0])

    t, y, u, e = Trapezoid_control(motor, cont)
    draw_control_result_chart(t,y,u,e, yr)


# In[ ]:


def demo1():
    interact(demo_sim, 
             kp=FloatSlider(0.1,min=0.01, max=2.0, step=0.01, continuous_update=False), 
             ki=FloatSlider(10,min=1, max=200, step=1, continuous_update=False),
             yr=FloatSlider(0.8,min=0.01, max=2, step=0.01, continuous_update=False)
            )


# In[ ]:


def demo2():
    kp=0.01
    ki=100.0

    sys=motor1724()
    cont=tf([kp, ki],[1, 0])

    tf_one=tf(1,1)
    sys_open=tf_multi(cont, sys)
    sys_den=tf_add( tf_one, sys_open )
    sys_y=tf_multi( sys_open, tf_inv(sys_den))
    sys_u=tf_multi(cont,  tf_inv(sys_den))
    sys_e=tf_multi(tf_one,  tf_inv(sys_den))

    starttime=0.0
    endtime=2.0
    steps=10000

    alpha1=2
    alpha2=4


    t,y0=ramp(sys_y, starttime, endtime, step=steps)

    y1=y0

    shift_time=0.4
    shift_value=int(len(y0)*shift_time/(endtime-starttime))
    y2=shift(y0, shift_value)

    shift_time=0.6
    shift_value=int(len(y0)*shift_time/(endtime-starttime))
    y3=shift(y0, shift_value)

    shift_time=0.8
    shift_value=int(len(y0)*shift_time/(endtime-starttime))
    y4=shift(y0, shift_value)

    y=2*(y1-y2)-4*(y3-y4)

    p1 = figure(plot_height=300, 
                plot_width=800,
                background_fill_color='#efefef',
                x_axis_label='Time(s)',
                y_axis_label='Omega(rad/s)'
    )
    p1.line(t, y, color="#ff0000", line_width=1.5, alpha=1.0)
    show(p1)


# In[ ]:


if __name__=='__main__':
    demo1()


# In[ ]:




