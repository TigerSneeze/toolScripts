clc; clear; close all;

fig = figure('Position', [100, 100, 1200, 600]);

%%%%%%%%%%%%% plotting debug %%%%%%%%%%%%%%%%%%%%%%
% MDPSO - 385628
load('ICRAF_convhist.mat') % Load your data
x = globalbestlog(end).var;
% 
outputVideo = VideoWriter('animation_sub_mdpso.mp4', 'MPEG-4');
titletext = sprintf("Robot Movement - Single-Objective Mixed-Discrete PSO \n(Optimal Symmetricity)");

% for failed case:
% outputVideo = VideoWriter('animation_sub_fail3.mp4', 'MPEG-4');
% titletext = sprintf("Robot Movement - Baseline");
% [IC, data1] = getSimout_fail(x);
% m_add = x(2)*2;


% % NSGA trad-off - 288972
% load("GA_fin_lab.mat")
% x = x(23,:);
% 
% outputVideo = VideoWriter('animation_sub_nsga_trade.mp4', 'MPEG-4');
% titletext = sprintf("Robot Movement - Multi-Objective GA \n(Optimal Trade-off Between Symetricity and Sensitivity)");


% outputVideo = VideoWriter('animation_sub_nsga_anchor.mp4', 'MPEG-4');
% titletext = sprintf("Robot Movement - Multi-Objective GA \n(Optimal Symmetricity)");
% % NSGA anchor - 423897
% x = x(33,:);

open(outputVideo);

[IC, data1] = getSimout(x); % Assume this function provides your initial conditions and data
m_add = x(2);

%%%%%%%%%%%%% plotting debug %%%%%%%%%%%%%%%%%%%%%%
% outputVideo = VideoWriter('test.mp4', 'MPEG-4');
% titletext = sprintf("test");
% open(outputVideo);
% load("debug_vars.mat")

%%%%%%%%%%%%% plotting debug END %%%%%%%%%%%%%%%%%%%%%%
[card, polar, touchdown_point, phase_flag_idx, hScatter, hLine, a] = initializeFig(IC, data1, m_add, titletext);

fly_flag = 0;
% Create the animation
for t = 1:1000:size(card, 1)
        
    if t >= phase_flag_idx(1)
        fly_flag = fly_flag + 1;
        phase_flag_idx(1) = [];
    end
    
    if mod(fly_flag, 2) == 1
        touchdown_point = [card(t, 1) + IC(1) * cos(polar(t, 1)), card(t, 4) + IC(1) * sin(polar(t, 1))];
    end

%     add annotation
    str = sprintf('Time: %.3f seconds', t/1e6);
    set(a,'String',str);
    a.FontSize = 20;

   % Spring setting
    xa = touchdown_point(1); xb = card(t,1); ya = touchdown_point(2); yb = card(t,4); ne = 5; a_spring= 0.02; ro = 0.002;
    [xs,ys] = spring(xa,ya,xb,yb,ne,a_spring,ro);

    set(hScatter, 'XData', card(t, 1), 'YData', card(t, 4));
%     set(hLine, 'XData', [touchdown_point(1), card(t, 1)], 'YData', [touchdown_point(2), card(t, 4)]);
    set(hLine, 'XData', [xs], 'YData', [ys]);
    
    drawnow;  % Update the figure
    currFrame = getframe(gcf);  % Capture current figure frame
    writeVideo(outputVideo, currFrame);  % Write the frame to the video file
end

close(outputVideo);
hold off;



function [card, polar, touchdown_point, phase_flag_idx, hScatter, hLine, a] = initializeFig(IC, data1, m_add, titletext)
    card = data1.rec_states;
    polar = data1.polar_states;

    % Define the static elements of the plot
    touchdown_point = [cos(IC(3)) * IC(1) + card(1,1), 0];
    phase_flag_idx = find(data1.switch==1);
    hold on
    % Initialize the scatter and line plot
    hScatter = scatter(card(1,1), card(1,4), 2000*m_add^2,'filled'); % 'filled' makes the scatter point solid
%     hLine = plot([touchdown_point(1), card(1,1)], [touchdown_point(2), card(1,4)],'LineWidth',3);
    xa = touchdown_point(1); xb = card(1,1); ya = touchdown_point(2); yb = card(1,4); ne = 5; a_spring= 0.02; ro = 0.002;
    [xs,ys] = spring(xa,ya,xb,yb,ne,a_spring,ro);
    hLine = plot(xs,ys,'LineWidth',2);

    bLine = plot(card(:,1), card(:,4), '--k','LineWidth',2);
    dim = [.2+0.25 .5 .3+0.25 .3];
    t = 0;
    str = sprintf('Time: %.3f seconds', t/1e6);
    a = annotation('textbox',dim,'String',str, 'FitBoxToText','on');
    a.FontSize = 20;
    % highlight stance phase
    counter = 0;
    head = 1;
    for i=phase_flag_idx'
        disp(['i = ', num2str(i)]);
        if mod(counter, 2)==0
            plot(card(head:i,1), card(head:i,4), '-r','LineWidth',2);
        end
        counter = counter + 1;
        head = i;
    end
    
    xl = xlabel('X-axis');
    yl = ylabel('Y-axis');
    ttl = title(titletext);


%     currentXTicks = xticks;
%     newXTicks = linspace(min(currentXTicks), max(currentXTicks), length(currentXTicks));
    newXTicks = linspace(0, 0.25, 11);
    xticks(newXTicks);

    font_s = 20;
    set(xl, 'FontSize', font_s)
    set(yl, 'FontSize', font_s)
    set(ttl, 'FontSize', font_s)
    grid minor;
    axis equal;

    ax = gca; % Get the current axes handle
    set(ax, 'FontSize', font_s-2);
    ax.XAxis(1).TickLabelFormat = '%.3f';
    ax.YAxis(1).TickLabelFormat = '%.2f';
    
    % For an animation, you might want to set consistent axis limits
    xlim([-0.02, 0.25]);
    ylim([0, 0.1]);

end
