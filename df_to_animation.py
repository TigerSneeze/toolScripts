import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import random
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

# full_df = pd.read_pickle('df_animate_4mu_nom.pkl')
# # savename = "animated_4mu_nominal_val_early_bar.mp4"

# full_df_R = pd.read_pickle('df_animate_4mu_RL.pkl')
# savename = "animated_4mu_RL_val_early_bar_COM.mp4"
# mu_num = 4

full_df = pd.read_pickle('df_animate_8mu_nom.pkl')
# savename = "animated_8mu_nominal_val_full_bar.mp4"


full_df_R = pd.read_pickle('df_animate_8mu_RL.pkl')
savename = "animated_8mu_RL_val_early_bar_COM.mp4"
mu_num = 8


# df_us = full_df.iloc[:-150]
df_us = full_df.iloc[:-150]
df_R = full_df_R.iloc[:-150]

if mu_num == 4:
    x1, y1, z1 = df_us['CM1_X'].values, df_us['CM1_Y'].values, df_us['CM1_Z'].values
    x2, y2, z2 = df_us['CM2_X'].values, df_us['CM2_Y'].values, df_us['CM2_Z'].values
    x3, y3, z3 = df_us['CM3_X'].values, df_us['CM3_Y'].values, df_us['CM3_Z'].values
    x4, y4, z4 = df_us['CM4_X'].values, df_us['CM4_Y'].values, df_us['CM4_Z'].values
    tx, ty, tz = df_us['TarX'].values, df_us['TarY'].values, -df_us['TarZ'].values # NEGATIVE HERE

    x1R, y1R, z1R = df_R['CM1_X'].values, df_R['CM1_Y'].values, df_R['CM1_Z'].values
    x2R, y2R, z2R = df_R['CM2_X'].values, df_R['CM2_Y'].values, df_R['CM2_Z'].values
    x3R, y3R, z3R = df_R['CM3_X'].values, df_R['CM3_Y'].values, df_R['CM3_Z'].values
    x4R, y4R, z4R = df_R['CM4_X'].values, df_R['CM4_Y'].values, df_R['CM4_Z'].values
    txR, tyR, tzR = df_R['TarX'].values, df_R['TarY'].values, -df_R['TarZ'].values # NEGATIVE HERE


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    fig = plt.figure(figsize=(18, 6))
    plt.subplots_adjust(left=0, right=0.95)
    ax = fig.add_subplot(131, projection='3d')
    ax.set_xlabel("X, m")
    ax.set_ylabel("Y, m")
    ax.set_zlabel("Z, m")


    line1, = ax.plot(x1[0:1], y1[0:1], z1[0:1], color='green', label='MU1')  # start with first point
    line2, = ax.plot(x2[0:1], y2[0:1], z2[0:1], color='violet', label='MU2')  # start with first point
    line3, = ax.plot(x3[0:1], y3[0:1], z3[0:1], color='red', label='MU3')  # start with first point
    line4, = ax.plot(x4[0:1], y4[0:1], z4[0:1], color='blue', label='MU4')  # start with first point
    ax.scatter(tx[0], ty[0], tz[0], marker='D', color='k', s=100)
    ax.legend(loc='upper right')
    ax.set_title("Meneuverable Units Trajectory\n(Nominal)")


    ax.set_xlim([min(min(x1), min(x2), min(x3), min(x4), min(tx)), max(max(x1), max(x2), max(x3), max(x4), max(tx))])
    ax.set_ylim([min(min(y1), min(y2), min(y3), min(y4), min(ty)), max(max(y1), max(y2), max(y3), max(y4), max(ty))])
    ax.set_zlim([min(min(z1), min(z2), min(z3), min(z4), min(tz)), max(max(z1), max(z2), max(z3), max(z4), max(tz))])


    # Second subplot for the bar plot
    ax2 = fig.add_subplot(132)
    pos = ax2.get_position()
    new_left = pos.x0 + 0.05  # Move 5% to the right
    ax2.set_position([new_left, pos.y0, pos.width, pos.height])
    bar_labels = ['Fuel1','Fuel1R', 'Fuel2','Fuel2R', 'Fuel3','Fuel3R', 'Fuel4', 'Fuel4R']
    bar_colors = ['green', 'green', 'violet','violet', 'red','red', 'blue', 'blue']
    bar_values = [df_us['Fuel1'][0], df_R['Fuel1'][0], df_us['Fuel2'][0], df_R['Fuel2'][0], df_us['Fuel3'][0], df_R['Fuel3'][0], df_us['Fuel4'][0], df_R['Fuel4'][0]]
    bars = ax2.bar(bar_labels, bar_values, color=bar_colors)

    for i, bar in enumerate(bars):
        if i % 2 == 1:
            bar.set_facecolor('none')
            bar.set_edgecolor(bar_colors[i])

    ax2.set_ylim([0, 0.035])
    ax2.set_ylabel("Fuel Consumption, kg")
    ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax2.set_xticklabels(['MU1', 'MU1_RL', 'MU2','MU2_RL', 'MU3','MU3_RL', 'MU4','MU4_RL'], rotation=45)
    ax2.set_title("Fuel Consumption\nNominal Solution: Solid bars     RL-based Solution: Hollow bars")

    # Initialize the Fuel_sum message text
    fuel_sum_text = ax2.text(0.5, 0.9, f"Net Fuel Consumption: {df_us['Fuel_sum'][0]:.5f}\nNet Fuel Consumption (RL): {df_R['Fuel_sum'][0]:.5f}", transform=ax2.transAxes, ha='center')
    fuel_sum_text.set_fontsize(16)
    plt.subplots_adjust(left=0, right=0.95)
    axR = fig.add_subplot(133, projection='3d')
    axR.set_xlabel("X, m")
    axR.set_ylabel("Y, m")
    axR.set_zlabel("Z, m")

    line1_R, = axR.plot(x1R[0:1], y1R[0:1], z1R[0:1], color='green', label='MU1R')  # start with first point
    line2_R, = axR.plot(x2R[0:1], y2R[0:1], z2R[0:1], color='violet', label='MU2R')  # start with first point
    line3_R, = axR.plot(x3R[0:1], y3R[0:1], z3R[0:1], color='red', label='MU3R')  # start with first point
    line4_R, = axR.plot(x4R[0:1], y4R[0:1], z4R[0:1], color='blue', label='MU4R')  # start with first point
    axR.scatter(txR[0], tyR[0], tzR[0], marker='D', color='k', s=100)
    axR.legend(loc='upper right')
    axR.set_title("Meneuverable Units Trajectory\n(Reinforcement Learning and PID)")


    axR.set_xlim([min(min(x1R), min(x2R), min(x3R), min(x4R), min(txR)), max(max(x1R), max(x2R), max(x3R), max(x4R), max(txR))])
    axR.set_ylim([min(min(y1R), min(y2R), min(y3R), min(y4R), min(tyR)), max(max(y1R), max(y2R), max(y3R), max(y4R), max(tyR))])
    axR.set_zlim([min(min(z1R), min(z2R), min(z3R), min(z4R), min(tzR)), max(max(z1R), max(z2R), max(z3R), max(z4R), max(tzR))])


    def animate(i):
        line1.set_data(x1[:i+1], y1[:i+1])  # update the data
        line1.set_3d_properties(z1[:i+1])  # for 3D plotting

        line2.set_data(x2[:i+1], y2[:i+1])  # update the data
        line2.set_3d_properties(z2[:i+1])  # for 3D plotting

        line3.set_data(x3[:i+1], y3[:i+1])  # update the data
        line3.set_3d_properties(z3[:i+1])  # for 3D plotting

        line4.set_data(x4[:i+1], y4[:i+1])  # update the data
        line4.set_3d_properties(z4[:i+1])  # for 3D plotting

        line1_R.set_data(x1R[:i+1], y1R[:i+1])  # update the data
        line1_R.set_3d_properties(z1R[:i+1])  # for 3D plotting

        line2_R.set_data(x2R[:i+1], y2R[:i+1])  # update the data
        line2_R.set_3d_properties(z2R[:i+1])  # for 3D plotting

        line3_R.set_data(x3R[:i+1], y3R[:i+1])  # update the data
        line3_R.set_3d_properties(z3R[:i+1])  # for 3D plotting

        line4_R.set_data(x4R[:i+1], y4R[:i+1])  # update the data
        line4_R.set_3d_properties(z4R[:i+1])  # for 3D plotting


        # Update the bar plot
        new_values = [df_us['Fuel1'][i],df_R['Fuel1'][i], df_us['Fuel2'][i],df_R['Fuel2'][i], df_us['Fuel3'][i], df_R['Fuel3'][i], df_us['Fuel4'][i], df_R['Fuel4'][i],]
        for bar, new_value in zip(bars, new_values):
            bar.set_height(new_value)
        
        # Update the Fuel_sum message
        # fuel_sum = df_us['Fuel_sum'].iloc[i]
        # fuel_sumR = df_R['Fuel_sum'].iloc[i]
        fuel_sum_text.set_text(f"Net Fuel Consumption: {df_us['Fuel_sum'][i]:.5f}\nNet Fuel Consumption (RL): {df_R['Fuel_sum'][i]:.5f}")
        fuel_sum_text.set_fontsize(16)
        # ax2.relim()  # Recalculate limits
        # ax2.autoscale_view(True, True, True)  # Update axes

        return line1, line2, line3, line4, line1_R, line2_R, line3_R, line4_R, bars, fuel_sum_text

    ani = animation.FuncAnimation(fig, animate, frames=range(len(x1)), interval=100, repeat=False)

    # Uncomment the next line if you want to save the animation as a .mp4 file
    ani.save(savename)

    plt.show()

elif mu_num==8:
    x1, y1, z1 = df_us['CM1_X'].values, df_us['CM1_Y'].values, df_us['CM1_Z'].values
    x2, y2, z2 = df_us['CM2_X'].values, df_us['CM2_Y'].values, df_us['CM2_Z'].values
    x3, y3, z3 = df_us['CM3_X'].values, df_us['CM3_Y'].values, df_us['CM3_Z'].values
    x4, y4, z4 = df_us['CM4_X'].values, df_us['CM4_Y'].values, df_us['CM4_Z'].values
    x5, y5, z5 = df_us['CM5_X'].values, df_us['CM5_Y'].values, df_us['CM5_Z'].values
    x6, y6, z6 = df_us['CM6_X'].values, df_us['CM6_Y'].values, df_us['CM6_Z'].values
    x7, y7, z7 = df_us['CM7_X'].values, df_us['CM7_Y'].values, df_us['CM7_Z'].values
    x8, y8, z8 = df_us['CM8_X'].values, df_us['CM8_Y'].values, df_us['CM8_Z'].values

    tx, ty, tz = df_us['TarX'].values, df_us['TarY'].values, -df_us['TarZ'].values

    x1R, y1R, z1R = df_R['CM1_X'].values, df_R['CM1_Y'].values, df_R['CM1_Z'].values
    x2R, y2R, z2R = df_R['CM2_X'].values, df_R['CM2_Y'].values, df_R['CM2_Z'].values
    x3R, y3R, z3R = df_R['CM3_X'].values, df_R['CM3_Y'].values, df_R['CM3_Z'].values
    x4R, y4R, z4R = df_R['CM4_X'].values, df_R['CM4_Y'].values, df_R['CM4_Z'].values
    x5R, y5R, z5R = df_R['CM5_X'].values, df_R['CM5_Y'].values, df_R['CM5_Z'].values
    x6R, y6R, z6R = df_R['CM6_X'].values, df_R['CM6_Y'].values, df_R['CM6_Z'].values
    x7R, y7R, z7R = df_R['CM7_X'].values, df_R['CM7_Y'].values, df_R['CM7_Z'].values
    x8R, y8R, z8R = df_R['CM8_X'].values, df_R['CM8_Y'].values, df_R['CM8_Z'].values
    txR, tyR, tzR = df_R['TarX'].values, df_R['TarY'].values, -df_R['TarZ'].values

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    fig = plt.figure(figsize=(15, 5))

    # Create a gridspec object
    gs = gridspec.GridSpec(1, 4)
    plt.subplots_adjust(left=0, right=0.95)
    ax = plt.subplot(gs[0, 0:1], projection='3d')
    ax.set_xlabel("X, m")
    ax.set_ylabel("Y, m")
    ax.text2D(0.99, 0.95, "Z, m", transform=ax.transAxes, rotation=90)


    line1, = ax.plot(x1[0:1], y1[0:1], z1[0:1], color='green', label='MU1')  # start with first point
    line2, = ax.plot(x2[0:1], y2[0:1], z2[0:1], color='violet', label='MU2')  # start with first point
    line3, = ax.plot(x3[0:1], y3[0:1], z3[0:1], color='red', label='MU3')  # start with first point
    line4, = ax.plot(x4[0:1], y4[0:1], z4[0:1], color='blue', label='MU4')  # start with first point
    line5, = ax.plot(x5[0:1], y5[0:1], z5[0:1], color='yellow', label='MU5')  # start with first point
    line6, = ax.plot(x6[0:1], y6[0:1], z6[0:1], color='cyan', label='MU6')  # start with first point
    line7, = ax.plot(x7[0:1], y7[0:1], z7[0:1], color='magenta', label='MU7')  # start with first point
    line8, = ax.plot(x8[0:1], y8[0:1], z8[0:1], color='black', label='MU8')  # start with first point

    ax.scatter(tx[0], ty[0], tz[0], marker='D', color='k', s=100)
    ax.legend(loc='upper right')
    ax.set_title("Meneuverable Units Trajectory\n(Nominal)")

    ax.set_xlim([min(min(x1), min(x2), min(x3), min(x4), min(tx)), max(max(x1), max(x2), max(x3), max(x4), max(tx))])
    ax.set_ylim([min(min(y1), min(y2), min(y3), min(y4), min(ty)), max(max(y1), max(y2), max(y3), max(y4), max(ty))])
    ax.set_zlim([min(min(z1), min(z2), min(z3), min(z4), min(tz)), max(max(z1), max(z2), max(z3), max(z4), max(tz))])


    # Second subplot for the bar plot
    
    ax2 = plt.subplot(gs[0, 1:3])
    pos = ax2.get_position()
    new_pos = [pos.x0 + 0.03, pos.y0,  pos.width, pos.height]  # Move to the right by 0.1
    ax2.set_position(new_pos)
    bar_labels = ['Fuel1', 'Fuel1R', 'Fuel2', 'Fuel2R', 'Fuel3', 'Fuel3R', 'Fuel4', 'Fuel4R', 'Fuel5', 'Fuel5R', 'Fuel6', 'Fuel6R', 'Fuel7', 'Fuel7R', 'Fuel8', 'Fuel8R']
    bar_colors = ['green', 'green', 'violet', 'violet', 'red', 'red', 'blue', 'blue', 'yellow', 'yellow', 'cyan', 'cyan', 'magenta', 'magenta', 'black', 'black']
    bar_values = [df_us['Fuel1'][0], df_R['Fuel1'][0], df_us['Fuel2'][0], df_R['Fuel2'][0], df_us['Fuel3'][0], df_R['Fuel3'][0], df_us['Fuel4'][0], df_R['Fuel4'][0], df_us['Fuel5'][0], df_R['Fuel5'][0], df_us['Fuel6'][0], df_R['Fuel6'][0], df_us['Fuel7'][0], df_R['Fuel7'][0], df_us['Fuel8'][0], df_R['Fuel8'][0]]

    bars = ax2.bar(bar_labels, bar_values, color=bar_colors)

    for i, bar in enumerate(bars):
        if i % 2 == 1:
            bar.set_facecolor('none')
            bar.set_edgecolor(bar_colors[i])
            bar.set_linewidth(2) 

    ax2.set_ylim([0, 0.035])
    ax2.set_ylabel("Fuel Consumption, kg")
    ax2.set_xticks(range(16))
    ax2.set_xticklabels(['MU1', 'MU1_RL', 'MU2', 'MU2_RL', 'MU3', 'MU3_RL', 'MU4', 'MU4_RL', 'MU5', 'MU5_RL', 'MU6', 'MU6_RL', 'MU7', 'MU7_RL', 'MU8', 'MU8_RL'], rotation=45)
    ax2.set_title("Fuel Consumption\n(Nominal Solution: Solid bars     RL-based Solution: Hollow bars)")
    # Initialize the Fuel_sum message text
    fuel_sum_text = ax2.text(0.5, 0.8, f"Net Fuel Consumption: {df_us['Fuel_sum'][0]:.5f}\nNet Fuel Consumption (RL): {df_R['Fuel_sum'][0]:.5f}", transform=ax2.transAxes, ha='center')
    fuel_sum_text.set_fontsize(16)
    
    axR = plt.subplot(gs[0, 3:4], projection='3d')
    axR.set_xlabel("X, m")
    axR.set_ylabel("Y, m")
    axR.set_title("Meneuverable Units Trajectory\n(Reinforcement Learning and PID)")
    axR.text2D(0.99, 0.95, "Z, m", transform=axR.transAxes, rotation=90)

    line1_R, = axR.plot(x1R[0:1], y1R[0:1], z1R[0:1], color='green', label='MU1R')  # start with first point
    line2_R, = axR.plot(x2R[0:1], y2R[0:1], z2R[0:1], color='violet', label='MU2R')  # start with first point
    line3_R, = axR.plot(x3R[0:1], y3R[0:1], z3R[0:1], color='red', label='MU3R')  # start with first point
    line4_R, = axR.plot(x4R[0:1], y4R[0:1], z4R[0:1], color='blue', label='MU4R')  # start with first point
    line5_R, = axR.plot(x5R[0:1], y5R[0:1], z5R[0:1], color='yellow', label='MU5R')  # start with first point
    line6_R, = axR.plot(x6R[0:1], y6R[0:1], z6R[0:1], color='cyan', label='MU6R')  # start with first point
    line7_R, = axR.plot(x7R[0:1], y7R[0:1], z7R[0:1], color='magenta', label='MU7R')  # start with first point
    line8_R, = axR.plot(x8R[0:1], y8R[0:1], z8R[0:1], color='black', label='MU8R')  # start with first point

    axR.scatter(txR[0], tyR[0], tzR[0], marker='D', color='k', s=100)
    axR.legend(loc='upper right')

    axR.set_xlim([min(min(x1R), min(x2R), min(x3R), min(x4R), min(txR)), max(max(x1R), max(x2R), max(x3R), max(x4R), max(txR))])
    axR.set_ylim([min(min(y1R), min(y2R), min(y3R), min(y4R), min(tyR)), max(max(y1R), max(y2R), max(y3R), max(y4R), max(tyR))])
    axR.set_zlim([min(min(z1R), min(z2R), min(z3R), min(z4R), min(tzR)), max(max(z1R), max(z2R), max(z3R), max(z4R), max(tzR))])

    def animate(i):
        line1.set_data(x1[:i+1], y1[:i+1])  # update the data
        line1.set_3d_properties(z1[:i+1])  # for 3D plotting

        line2.set_data(x2[:i+1], y2[:i+1])  # update the data
        line2.set_3d_properties(z2[:i+1])  # for 3D plotting

        line3.set_data(x3[:i+1], y3[:i+1])  # update the data
        line3.set_3d_properties(z3[:i+1])  # for 3D plotting

        line4.set_data(x4[:i+1], y4[:i+1])  # update the data
        line4.set_3d_properties(z4[:i+1])  # for 3D plotting

        line5.set_data(x5[:i+1], y5[:i+1])  # update the data
        line5.set_3d_properties(z5[:i+1])  # for 3D plotting

        line6.set_data(x6[:i+1], y6[:i+1])  # update the data
        line6.set_3d_properties(z6[:i+1])  # for 3D plotting

        line7.set_data(x7[:i+1], y7[:i+1])  # update the data
        line7.set_3d_properties(z7[:i+1])  # for 3D plotting

        line8.set_data(x8[:i+1], y8[:i+1])  # update the data
        line8.set_3d_properties(z8[:i+1])  # for 3D plotting

        # Add the "_R" lines
        line1_R.set_data(x1R[:i+1], y1R[:i+1])  # update the data
        line1_R.set_3d_properties(z1R[:i+1])  # for 3D plotting

        line2_R.set_data(x2R[:i+1], y2R[:i+1])  # update the data
        line2_R.set_3d_properties(z2R[:i+1])  # for 3D plotting

        line3_R.set_data(x3R[:i+1], y3R[:i+1])  # update the data
        line3_R.set_3d_properties(z3R[:i+1])  # for 3D plotting

        line4_R.set_data(x4R[:i+1], y4R[:i+1])  # update the data
        line4_R.set_3d_properties(z4R[:i+1])  # for 3D plotting

        line5_R.set_data(x5R[:i+1], y5R[:i+1])  # update the data
        line5_R.set_3d_properties(z5R[:i+1])  # for 3D plotting

        line6_R.set_data(x6R[:i+1], y6R[:i+1])  # update the data
        line6_R.set_3d_properties(z6R[:i+1])  # for 3D plotting

        line7_R.set_data(x7R[:i+1], y7R[:i+1])  # update the data
        line7_R.set_3d_properties(z7R[:i+1])  # for 3D plotting

        line8_R.set_data(x8R[:i+1], y8R[:i+1])  # update the data
        line8_R.set_3d_properties(z8R[:i+1])  # for 3D plotting

        # Update the bar plot
        new_values = [
            df_us['Fuel1'][i], df_R['Fuel1'][i], 
            df_us['Fuel2'][i], df_R['Fuel2'][i], 
            df_us['Fuel3'][i], df_R['Fuel3'][i], 
            df_us['Fuel4'][i], df_R['Fuel4'][i],
            df_us['Fuel5'][i], df_R['Fuel5'][i],
            df_us['Fuel6'][i], df_R['Fuel6'][i],
            df_us['Fuel7'][i], df_R['Fuel7'][i],
            df_us['Fuel8'][i], df_R['Fuel8'][i]
        ]
        for bar, new_value in zip(bars, new_values):
            bar.set_height(new_value)
        
        # Update the Fuel_sum message
        fuel_sum_text.set_text(f"Net Fuel Consumption: {df_us['Fuel_sum'][i]:.5f}\nNet Fuel Consumption (RL): {df_R['Fuel_sum'][i]:.5f}")
        fuel_sum_text.set_fontsize(16)
        # ax2.relim()  # Recalculate limits
        # ax2.autoscale_view(True, True, True)  # Update axes

        return line1, line2, line3, line4, line5, line6, line7, line8, line1_R, line2_R, line3_R, line4_R, line5_R, line6_R, line7_R, line8_R, bars, fuel_sum_text


    ani = animation.FuncAnimation(fig, animate, frames=range(len(x1)), interval=100, repeat=False)

    # Uncomment the next line if you want to save the animation as a .mp4 file
    ani.save(savename)

    plt.show()