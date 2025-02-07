%% calculation for MPC experiments
clear all
close all
clc
% cd /home/robot/workspaces/Big_Data/Tests/walk/MPC/AB/20230305_155054
iter_total_file = 0;
max_mpc_time = zeros(1,50);
average_mpc_time = zeros(1,50);
eucl_norm_table = zeros(50,1);
error_mpc = zeros(50,1);
tot_len = zeros(50,1);
for file_number =1:25
    iter_total_file = iter_total_file+1;
    fileneme = sprintf('%i.mat',file_number);
    load(fileneme);
    mpc_time = low_mpc_details(:,2);
    len_f = length(joint_positions);
    tot_len(file_number) = len_f;
    max_mpc_time(file_number) = max(mpc_time);
    average_mpc_time(file_number)=sum(mpc_time)/len_f;
    high_goal = from_high_controller(:,19:24);
    s_mpc = 1;
    
    t_mpc = len_f-s_mpc;
    cost = cost_calc(joint_positions(s_mpc:len_f,:), high_goal(s_mpc:len_f,:), mpc_sol(s_mpc:len_f,1:6), human_poses(s_mpc:len_f,:), t_mpc);
    mpc_cost = cost(1,:);
    mpc_jp_cost = cost(2,:);
    mpc_jv_cost = cost(3,:);
    mpc_rh_cost = cost(4,:);
    jp = joint_positions;
    jv = mpc_sol(:,1:6);
    dt = 0.05;
    A = zeros(len_f,14);
    error_mpc(file_number) = 0;
    for f=1:len_f
        tp = test_points(jp(f,1),jp(f,2),jp(f,3),jp(f,4),jp(f,5),jp(f,6));
        tp_vel = test_vel(jp(f,1),jp(f,2),jp(f,3),jp(f,4),jp(f,5),jp(f,6),jv(f,1),jv(f,2),jv(f,3),jv(f,4),jv(f,5),jv(f,6));
        A(f,:) = limit_check(tp, human_poses(f,:), tp_vel);
    end
    
    name = sprintf('ssm_%i.png', file_number)
    plot_2f(ctv_linear(:,1:7), lin_vel_limit(:,1:7),  name, dt, s_mpc, len_f,"lin vel","limit")
    ee_pose_list = zeros(len_f,3);
    
    for ii=1:len_f
        temp = joint_positions(ii,:);
        ee_pose_list(ii,:) = ee_pose(temp(1),temp(2),temp(3),temp(4),temp(5),temp(6));
    end

    eucl_norm = zeros(len_f-1,1);
    for ii=1:len_f-1
        a = ee_pose_list(ii);
        b = ee_pose_list(ii+1);
        eucl_norm(ii) = norm(b-a);
    end
    eucl_norm_table(file_number) = sum(eucl_norm);
    
    time_spent(file_number) = time(len_f)-time(s_mpc);
    total_MPC(file_number) = sum(mpc_cost);
    total_MPC_jp(file_number) = sum(mpc_jp_cost);
    total_MPC_jv(file_number) = sum(mpc_jv_cost);
    total_MPC_rh(file_number) = sum(mpc_rh_cost);
end
% big_time = max(time_spent)
maximum_solver_time = max(max_mpc_time)
average_solver_time = sum(average_mpc_time)/iter_total_file
average_time = sum(time_spent)/iter_total_file
average_total_cost = sum(total_MPC)/iter_total_file
% average_jp_cost = sum(total_MPC_jp)/iter_total_file
average_jv_cost = sum(total_MPC_jv)/iter_total_file
average_rh_cost = sum(total_MPC_rh)/iter_total_file
average_e = sum(eucl_norm_table)/iter_total_file
% error_av = sum(error_mpc)/sum(tot_len)
%% calculation for NN files
clear all
close all
clc
% cd /home/robot/workspaces/Big_Data/Tests/DNN/walk/AB/20230304_184650
gg = 25
iter_total_file = 0;
max_nn_time = zeros(1,gg);
eucl_norm_table = zeros(gg,1);
average_nn_time = zeros(1,gg);
lens = zeros(gg,1);
error_nn_pr = zeros(gg,1);
for file_number = 1:gg
    iter_total_file = iter_total_file+1;
    filename = sprintf('%i.mat', file_number);
    load(filename);
    max_nn_time(file_number) = max(nn_time);
    len = length(actions);
    average_nn_time(file_number)=sum(nn_time)/len;
%     robot_spheres = [0.15, 0.15, 0.15, 0.08, 0.08, 0.12, 0.1];
%     human_sphers = [0.5510,0.6010,0.5010,0.5010,0.5010,0.5010,0.5010,0.5010,0.4510,0.4510,0.4810,0.4810,0.5510,0.6010];
    s = 1;
    lens(file_number)=len;
    dt = 0.05;
    A = zeros(len,14);
    error_nn_pr(file_number) = 0;
    jp = joint_positions;
    jv = cas_vel;
    for f=1:len
        tp = test_points(jp(f,1),jp(f,2),jp(f,3),jp(f,4),jp(f,5),jp(f,6));
        tp_vel = test_vel(jp(f,1),jp(f,2),jp(f,3),jp(f,4),jp(f,5),jp(f,6),jv(f,1),jv(f,2),jv(f,3),jv(f,4),jv(f,5),jv(f,6));
        A(f,:) = limit_check(tp, human_poses(f,:), tp_vel);
    end
    
    name = sprintf('ssm_%i.png', file_number)
    plot_3f(nn_lin_vel(:,1:7), cas_lin_vel(:,1:7), limits(:,1:7), name, dt, s, len)
    ee_pose_list = zeros(len,3);
    name = sprintf('ssm_%i.png', file_number)
    
    for ii=1:len
        temp = joint_positions(ii,:);
        ee_pose_list(ii,:) = ee_pose(temp(1),temp(2),temp(3),temp(4),temp(5),temp(6));
    end

    eucl_norm = zeros(len-1,1);
    for ii=1:len-19.
        a = ee_pose_list(ii);
        b = ee_pose_list(ii+1);
        eucl_norm(ii) = norm(b-a);
    end
    eucl_norm_table(file_number) = sum(eucl_norm);
    max(nn_time);
    [len,l] = size(actions);
    e = len;
    s_nn = 3;
    t_nn = len-s_nn;

    cost = cost_calc(joint_positions(s_nn:e,:), goal(s_nn:e,:), cas_vel(s_nn:e,:), human_poses(s_nn:e,:), t_nn);
    nn_cost = cost(1,:);
    jp_cost = cost(2,:);
    jv_cost = cost(3,:);
    rh_cost = cost(4,:);
    total_NN(file_number) = sum(nn_cost);
    total_NN_jp(file_number) = sum(jp_cost);
    total_NN_jv(file_number) = sum(jv_cost);
    total_NN_rh(file_number) = sum(rh_cost);
    time_spent(file_number) = time(t_nn)-time(s_nn);

end
max_solver_time = max(max_nn_time)
average_solver_time = sum(average_nn_time)/iter_total_file
average_time = sum(time_spent)/iter_total_file
average_total_cost = sum(total_NN)/iter_total_file
% average_jp_cost = sum(total_NN_jp)/iter_total_file
average_jv_cost = sum(total_NN_jv)/iter_total_file
% average_rh_cost = sum(total_NN_rh)/iter_total_file
average_e = sum(eucl_norm_table)/iter_total_file
% error_av = sum(error_nn_pr)/sum(lens)
%% cost calculation
function C = cost_calc(jp, goal, q_d, human, l)
    gamma = 3;
    c1 = 10;
    c2 = 1;
    c3 = 500;
    for i = 1:l
        q = jp(i,1:6);
        q_goal = goal(i,1:6);
        q_dot = q_d(i,1:6);
        jp_cost(i) = norm(q-q_goal)*norm(q-q_goal);
        jp_cost(i) = jp_cost(i) * c1;
        jv_cost(i) = sum(q_dot.*q_dot);
        jv_cost(i) = jv_cost(i) * c2;
        Sphere1_X = human(i,1);
        Sphere1_Y = human(i,2);
        Sphere1_Z = human(i,3);
        Sphere2_X = human(i,4);
        Sphere2_Y = human(i,5);
        Sphere2_Z = human(i,6);
        Sphere14_X = human(i,19);
        Sphere14_Y = human(i,20);
        Sphere14_Z = human(i,21);

        human_stick(1) = (Sphere1_X + Sphere2_X + Sphere14_X)/3;
        human_stick(2) = (Sphere1_Y + Sphere2_Y + Sphere14_Y)/3;
        human_stick(3) = (Sphere1_Z + Sphere2_Z + Sphere14_Z)/3;
        ee_current_pose = ee_pose(q(1),q(2),q(3),q(4),q(5),q(6));
        ee_goal_pose = ee_pose(q_goal(1),q_goal(2),q_goal(3),q_goal(4),q_goal(5),q_goal(6));
        goal_dist = norm(ee_current_pose-ee_goal_pose);
        robot_obst_dist = norm(ee_current_pose-human_stick);
        RH_cost(i) = exp(-gamma*(robot_obst_dist/goal_dist));
        RH_cost(i) = RH_cost(i) * c3;
        Cost(i) = jp_cost(i)+jv_cost(i)+RH_cost(i);
    end
    C(1,:) = Cost;
    C(2,:) = jp_cost;
    C(3,:) = jv_cost;
    C(4,:) = RH_cost;
end

function A = ee_pose(q1,q2,q3,q4,q5,q6)
    cq1 = cos(q1);
	sq1 = sin(q1);
	cq2 = cos(q2);
	sq2 = sin(q2);
	cq3 = cos(q3);
	sq3 = sin(q3);
	cq4 = cos(q4);
	sq4 = sin(q4);
	cq5 = cos(q5);
	sq5 = sin(q5);
    cq6 = cos(q6);
	sq6 = sin(q6);
    
    T7_1 = 0.10915*sq1 - 0.425*cq1*cq2 + 0.0823*cq5*sq1 + 0.39225*cq1*sq2*sq3 - 0.0823*cos(q2 + q3 + q4)*cq1*sq5 + 0.09465*cos(q2 + q3)*cq1*sq4 + 0.09465*sin(q2 + q3)*cq1*cq4 - 0.39225*cq1*cq2*cq3-0.05*sq1;
    T7_2 = 0.39225*sq1*sq2*sq3 - 0.0823*cq1*cq5 - 0.425*cq2*sq1 - 0.10915*cq1 - 0.0823*cos(q2 + q3 + q4)*sq1*sq5 + 0.09465*cos(q2 + q3)*sq1*sq4 + 0.09465*sin(q2 + q3)*cq4*sq1 - 0.39225*cq2*cq3*sq1+0.05*cq1;
    T7_3 = 0.1+0.09465*sin(q2 + q3)*sq4 - 0.425*sq2 - 0.39225*sin(q2 + q3) - sq5*(0.0823*cos(q2 + q3)*sq4 + 0.0823*sin(q2 + q3)*cq4) - 0.09465*cos(q2 + q3)*cq4 + 0.08945;
    A = [T7_1,T7_2,T7_3];
end

function tp=test_points(theta_1,theta_2,theta_3,theta_4,theta_5,theta_6)
%     tp = zeros(7,3);
    z_sh = 0.1;
    tp(1) = 0.06*sin(theta_1);
    tp(2) = -0.06*cos(theta_1);
    tp(3) = 0.0894+z_sh;
    tp(4) = (-0.425*cos(theta_1)*cos(theta_2))/2+0.14*sin(theta_1);
    tp(5) = (-0.425*cos(theta_2)*sin(theta_1))/2-0.14*cos(theta_1);
    tp(6) = (0.0894 - 0.425*sin(theta_2))/2+z_sh;
    tp(7) = -0.425*cos(theta_1)*cos(theta_2)+0.11*sin(theta_1);
    tp(8) = -0.425*cos(theta_2)*sin(theta_1)-0.11*cos(theta_1);
    tp(9) = 0.0894 - 0.425*sin(theta_2)+z_sh;
    tp(10) = -0.425*cos(theta_1)*cos(theta_2)+(-(cos(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-(-0.425*cos(theta_1)*cos(theta_2)))/3+0.02*sin(theta_1);
    tp(11) = -0.425*cos(theta_2)*sin(theta_1)+(-(sin(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-(-0.425*cos(theta_2)*sin(theta_1)))/3-0.02*cos(theta_1);
    tp(12) = 0.0894 - 0.425*sin(theta_2)+(0.0894 - 0.425*sin(theta_2) - 0.39225*sin(theta_2 + theta_3)-(0.0894 - 0.425*sin(theta_2)))/3+z_sh;
    tp(13) = -0.425*cos(theta_1)*cos(theta_2)+2*(-(cos(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-(-0.425*cos(theta_1)*cos(theta_2)))/3+0.02*sin(theta_1);
    tp(14) = -0.425*cos(theta_2)*sin(theta_1)+2*(-(sin(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-(-0.425*cos(theta_2)*sin(theta_1)))/3-0.02*cos(theta_1);
    tp(15) = 0.0894 - 0.425*sin(theta_2)+2*(0.0894 - 0.425*sin(theta_2) - 0.39225*sin(theta_2 + theta_3)-(0.0894 - 0.425*sin(theta_2)))/3+z_sh;
    tp(16) = -(cos(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000+0.06*sin(theta_1);
    tp(17) = -(sin(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-0.06*cos(theta_1);
    tp(18) = 0.0894 - 0.425*sin(theta_2) - 0.39225*sin(theta_2 + theta_3)+z_sh;
    tp(19) = 0.10915*sin(theta_1) - 0.425*cos(theta_1)*cos(theta_2) + 0.0823*cos(theta_5)*sin(theta_1) + 0.39225*cos(theta_1)*sin(theta_2)*sin(theta_3) - 0.0823*cos(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) + 0.09465*cos(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) + 0.09465*sin(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) - 0.39225*cos(theta_1)*cos(theta_2)*cos(theta_3)-0.05*sin(theta_1);
    tp(20) = 0.39225*sin(theta_1)*sin(theta_2)*sin(theta_3) - 0.0823*cos(theta_1)*cos(theta_5) - 0.425*cos(theta_2)*sin(theta_1) - 0.10915*cos(theta_1) - 0.0823*cos(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.09465*cos(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) + 0.09465*sin(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) - 0.39225*cos(theta_2)*cos(theta_3)*sin(theta_1)+0.05*cos(theta_1);
    tp(21) = 0.09465*sin(theta_2 + theta_3)*sin(theta_4) - 0.425*sin(theta_2) - 0.39225*sin(theta_2 + theta_3) - sin(theta_5)*(0.0823*cos(theta_2 + theta_3)*sin(theta_4) + 0.0823*sin(theta_2 + theta_3)*cos(theta_4)) - 0.09465*cos(theta_2 + theta_3)*cos(theta_4) + 0.08945+z_sh;
 
end

function a = plot_f(data, name, dt, s, len)
    
    fig_1 = figure('Name', name);
    subplot(3,2,1);
    grid on;
    hold on;
    plot(data(s:len,1));
    % plot(lin_vell_limit_array(s:len,1));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("q 1")

    subplot(3,2,2);
    grid on;
    hold on;
    plot(data(s:len,2));
    % plot(lin_vell_limit_array(s:len,2));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("q 2 ")

    subplot(3,2,3);
    grid on;
    hold on;
    plot(data(s:len,3));
    % plot(lin_vell_limit_array(s:len,3));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("q 3")

    subplot(3,2,4);
    grid on;
    hold on;
    plot(data(s:len,4));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("q 4")

    subplot(3,2,5);
    grid on;
    hold on;
    plot(data(s:len,5));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("q 5")

    subplot(3,2,6);
    grid on;
    hold on;
    l1 = plot(data(s:len,6));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("q 6")
    saveas(fig_1,name)
end

function a = plot_2f(data1, data2, name, dt, s, len, bb,cc)
    
    fig_5 = figure('Name', name);
    subplot(4,2,1);
    grid on;
    hold on;
    plot(data1(s:len,1));
    plot(data2(s:len,1));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 1")

    subplot(4,2,2);
    grid on;
    hold on;
    plot(data1(s:len,2));
    plot(data2(s:len,2));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 2 ")

    subplot(4,2,3);
    grid on;
    hold on;
    plot(data1(s:len,3));
    plot(data2(s:len,3));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 3")

    subplot(4,2,4);
    grid on;
    hold on;
    plot(data1(s:len,4));
    plot(data2(s:len,4));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 4")

    subplot(4,2,5);
    grid on;
    hold on;
    plot(data1(s:len,5));
    plot(data2(s:len,5));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 5")
    
    subplot(4,2,6);
    grid on;
    hold on;
    plot(data1(s:len,6));
    plot(data2(s:len,6));
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 6")
    
    subplot(4,2,7);
    grid on;
    hold on;
    l1 = plot(data1(s:len,7));
    l2 = plot(data2(s:len,7));
    hL = legend([l1,l2],[bb, cc]);
    
    newPosition = [0.6 0.1 0.1 0.1];
    newUnits = 'normalized';
    set(hL,'Position', newPosition,'Units', newUnits);
    title("tp 7")
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
%     name = sprintf('limits_%i.png');
    saveas(fig_5, name);
end

function a = plot_3f(data1, data2, data3, name, dt, s, len)
    
    fig_5 = figure('Name', name);
    subplot(4,2,1);
    grid on;
    hold on;
    plot(data1(s:len,1),'b');
    plot(data2(s:len,1),'k');
    plot(data3(s:len,1),'r');
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 1")

    subplot(4,2,2);
    grid on;
    hold on;
    plot(data1(s:len,2),'b');
    plot(data2(s:len,2),'k');
    plot(data3(s:len,2),'r');
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 2 ")

    subplot(4,2,3);
    grid on;
    hold on;
    plot(data1(s:len,3),'b');
    plot(data2(s:len,3),'k');
    plot(data3(s:len,3),'r');
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 3")

    subplot(4,2,4);
    grid on;
    hold on;
    plot(data1(s:len,4),'b');
    plot(data2(s:len,4),'k');
    plot(data3(s:len,4),'r');
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 4")

    subplot(4,2,5);
    grid on;
    hold on;
    plot(data1(s:len,5),'b');
    plot(data2(s:len,5),'k');
    plot(data3(s:len,5),'r');
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 5")
    
    subplot(4,2,6);
    grid on;
    hold on;
    plot(data1(s:len,6),'b');
    plot(data2(s:len,6),'k');
    plot(data3(s:len,6),'r');
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 6")
    
    subplot(4,2,7);
    grid on;
    hold on;
    l1 = plot(data1(s:len,7),'b');
    l2 = plot(data2(s:len,7),'k');
    l3 = plot(data3(s:len,7),'r');
    set(gca,'XTick',0:100:100*len);
    set(gca,'XTickLabel',0:dt*100:len*100*dt);
    title("tp 7")
    hL = legend([l1,l2,l3],["nn output", "casadi output", "limit"]);
    newPosition = [0.6 0.1 0.1 0.1];
    newUnits = 'normalized';
    set(hL,'Position', newPosition,'Units', newUnits);
%     name = sprintf('limits_%i.png');
    saveas(fig_5, name);
end

function array = limit_check(test_point_cposes, human_poses, test_point_vels)
    array = zeros(1,14);
    robot_spheres = [0.15, 0.15, 0.15, 0.08, 0.08, 0.12, 0.1];
    sphere_radi = [0.5510,0.6010,0.5010,0.5010,0.5010,0.5010,0.5010,0.5010,0.4510,0.4510,0.4810,0.4810,0.5510,0.6010];
    min_dist = [1000,1000,1000,1000,1000,1000,1000];
    for j = 0:6
        w = [test_point_cposes(j*3+1),test_point_cposes(j*3+2),test_point_cposes(j*3+3)];
        for k = 0:13
            p = [human_poses(k*3+1),human_poses(k*3+2),human_poses(k*3+3)];
            local_val = norm(p-w);
            if local_val<min_dist(j+1)
                min_dist(j+1) = local_val;
                spheres_dist(j+1) = robot_spheres(j+1)+sphere_radi(k+1);
            end
        end
    end
    max_vell = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    max_linear_vell = 0;
    for k = 0:6
        temp_linear_vell = sqrt(test_point_vels(k*3+1)*test_point_vels(k*3+1)+test_point_vels(k*3+2)*test_point_vels(k*3+2)+test_point_vels(k*3+3)*test_point_vels(k*3+3));
        max_vell(k+1) = temp_linear_vell;
    end
    lin_vell_limit_arr=[10, 10, 10, 10, 10, 10, 10];
    alpha=[2.79, 1.95, 1, 0.8, 0.65, 0.45, 0.35];
    d_bar = 0.15;
    for i = 0:6
        sqrt_temp_value = (min_dist(i+1)+spheres_dist(i+1))*(min_dist(i+1)+spheres_dist(i+1))-spheres_dist(i+1)*spheres_dist(i+1);
        if sqrt_temp_value<0
           lin_vell_limit_arr(i+1) = 0.00000000000000;
        else
            lin_vell_limit_arr(i+1) = alpha(i+1)*sqrt(sqrt_temp_value);
        end
    end
    array(1:7) = lin_vell_limit_arr;
    array(8:14) = max_vell;
end

function A = test_vel(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, u_1,u_2,u_3,u_4,u_5,u_6)
    A = zeros(21,1);
    A(1)=0.06*u_1*cos(theta_1);
    A(2)=0.06*u_1*sin(theta_1);
    A(3)=0;
    A(4)=u_1*(0.14*cos(theta_1) + 0.2125*cos(theta_2)*sin(theta_1)) + 0.2125*u_2*cos(theta_1)*sin(theta_2);
    A(5)=u_1*(0.14*sin(theta_1) - 0.2125*cos(theta_1)*cos(theta_2)) + 0.2125*u_2*sin(theta_1)*sin(theta_2);
    A(6)=-0.2125*u_2*cos(theta_2);
    A(7)=u_1*(0.11*cos(theta_1) + 0.425*cos(theta_2)*sin(theta_1)) + 0.425*u_2*cos(theta_1)*sin(theta_2);
    A(8)=u_1*(0.11*sin(theta_1) - 0.425*cos(theta_1)*cos(theta_2)) + 0.425*u_2*sin(theta_1)*sin(theta_2);
    A(9)=-0.425*u_2*cos(theta_2);
    A(10)=0.02*u_1*cos(theta_1) + 0.13075*u_1*cos(theta_2 + theta_3)*sin(theta_1) + 0.13075*u_2*sin(theta_2 + theta_3)*cos(theta_1) + 0.13075*u_3*sin(theta_2 + theta_3)*cos(theta_1) + 0.425*u_1*cos(theta_2)*sin(theta_1) + 0.425*u_2*cos(theta_1)*sin(theta_2);
    A(11)=0.02*u_1*sin(theta_1) - 0.13075*u_1*cos(theta_2 + theta_3)*cos(theta_1) + 0.13075*u_2*sin(theta_2 + theta_3)*sin(theta_1) + 0.13075*u_3*sin(theta_2 + theta_3)*sin(theta_1) - 0.425*u_1*cos(theta_1)*cos(theta_2) + 0.425*u_2*sin(theta_1)*sin(theta_2);
    A(12)=- 1.0*u_2*(0.13075*cos(theta_2 + theta_3) + 0.425*cos(theta_2)) - 0.13075*u_3*cos(theta_2 + theta_3);
    A(13)=0.02*u_1*cos(theta_1) + 0.2615*u_1*cos(theta_2 + theta_3)*sin(theta_1) + 0.2615*u_2*sin(theta_2 + theta_3)*cos(theta_1) + 0.2615*u_3*sin(theta_2 + theta_3)*cos(theta_1) + 0.425*u_1*cos(theta_2)*sin(theta_1) + 0.425*u_2*cos(theta_1)*sin(theta_2);
    A(14)=0.02*u_1*sin(theta_1) - 0.2615*u_1*cos(theta_2 + theta_3)*cos(theta_1) + 0.2615*u_2*sin(theta_2 + theta_3)*sin(theta_1) + 0.2615*u_3*sin(theta_2 + theta_3)*sin(theta_1) - 0.425*u_1*cos(theta_1)*cos(theta_2) + 0.425*u_2*sin(theta_1)*sin(theta_2);
    A(15)=- 1.0*u_2*(0.2615*cos(theta_2 + theta_3) + 0.425*cos(theta_2)) - 0.2615*u_3*cos(theta_2 + theta_3);
    A(16)=u_1*(0.06*cos(theta_1) + 0.00025*sin(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2))) + 0.00025*u_2*cos(theta_1)*(1569.0*sin(theta_2 + theta_3) + 1700.0*sin(theta_2)) + 0.39225*u_3*sin(theta_2 + theta_3)*cos(theta_1);
    A(17)=u_1*(0.06*sin(theta_1) - 0.00025*cos(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2))) + 0.00025*u_2*sin(theta_1)*(1569.0*sin(theta_2 + theta_3) + 1700.0*sin(theta_2)) + 0.39225*u_3*sin(theta_2 + theta_3)*sin(theta_1);
    A(18)=- 1.0*u_2*(0.39225*cos(theta_2 + theta_3) + 0.425*cos(theta_2)) - 0.39225*u_3*cos(theta_2 + theta_3);
    A(19)=0.05915*u_1*cos(theta_1) + 0.0823*u_1*cos(theta_1)*cos(theta_5) + 0.425*u_1*cos(theta_2)*sin(theta_1) + 0.425*u_2*cos(theta_1)*sin(theta_2) - 0.0823*u_5*sin(theta_1)*sin(theta_5) + 0.09465*u_2*cos(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) + 0.09465*u_3*cos(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) + 0.09465*u_4*cos(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) - 0.09465*u_1*cos(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) - 0.09465*u_1*sin(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) - 0.09465*u_2*sin(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) - 0.09465*u_3*sin(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) - 0.09465*u_4*sin(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) + 0.39225*u_1*cos(theta_2)*cos(theta_3)*sin(theta_1) + 0.39225*u_2*cos(theta_1)*cos(theta_2)*sin(theta_3) + 0.39225*u_2*cos(theta_1)*cos(theta_3)*sin(theta_2) + 0.39225*u_3*cos(theta_1)*cos(theta_2)*sin(theta_3) + 0.39225*u_3*cos(theta_1)*cos(theta_3)*sin(theta_2) - 0.39225*u_1*sin(theta_1)*sin(theta_2)*sin(theta_3) - 0.0823*u_5*cos(theta_2 + theta_3 + theta_4)*cos(theta_1)*cos(theta_5) + 0.0823*u_1*cos(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.0823*u_2*sin(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) + 0.0823*u_3*sin(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) + 0.0823*u_4*sin(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5);
    A(20)=0.05915*u_1*sin(theta_1) - 0.425*u_1*cos(theta_1)*cos(theta_2) + 0.0823*u_1*cos(theta_5)*sin(theta_1) + 0.0823*u_5*cos(theta_1)*sin(theta_5) + 0.425*u_2*sin(theta_1)*sin(theta_2) + 0.09465*u_1*cos(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) + 0.09465*u_1*sin(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) + 0.09465*u_2*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) + 0.09465*u_3*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) + 0.09465*u_4*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) - 0.39225*u_1*cos(theta_1)*cos(theta_2)*cos(theta_3) - 0.09465*u_2*sin(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) - 0.09465*u_3*sin(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) - 0.09465*u_4*sin(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) + 0.39225*u_1*cos(theta_1)*sin(theta_2)*sin(theta_3) + 0.39225*u_2*cos(theta_2)*sin(theta_1)*sin(theta_3) + 0.39225*u_2*cos(theta_3)*sin(theta_1)*sin(theta_2) + 0.39225*u_3*cos(theta_2)*sin(theta_1)*sin(theta_3) + 0.39225*u_3*cos(theta_3)*sin(theta_1)*sin(theta_2) - 0.0823*u_1*cos(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) - 0.0823*u_5*cos(theta_2 + theta_3 + theta_4)*cos(theta_5)*sin(theta_1) + 0.0823*u_2*sin(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.0823*u_3*sin(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.0823*u_4*sin(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5);
    A(21)=u_4*(0.09465*sin(theta_2 + theta_3 + theta_4) - 0.0823*cos(theta_2 + theta_3 + theta_4)*sin(theta_5)) - 1.0*u_3*(0.39225*cos(theta_2 + theta_3) - 0.09465*sin(theta_2 + theta_3 + theta_4) + 0.0823*cos(theta_2 + theta_3 + theta_4)*sin(theta_5)) - 1.0*u_2*(0.39225*cos(theta_2 + theta_3) + 0.425*cos(theta_2) - 0.09465*cos(theta_2 + theta_3)*sin(theta_4) - 0.09465*sin(theta_2 + theta_3)*cos(theta_4) + 0.0823*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_5) - 0.0823*sin(theta_2 + theta_3)*sin(theta_4)*sin(theta_5)) - 0.0823*u_5*sin(theta_2 + theta_3 + theta_4)*cos(theta_5);

%     A(1) = 0.06*u_1*cos(theta_1);
%     A(2) = 0.06*u_1*sin(theta_1);
%     A(3) = 0;
%     A(4) = u_1*(0.14*cos(theta_1) + 0.2125*cos(theta_2)*sin(theta_1)) + 0.2125*u_2*cos(theta_1)*sin(theta_2);
%     A(5) = u_1*(0.14*sin(theta_1) - 0.2125*cos(theta_1)*cos(theta_2)) + 0.2125*u_2*sin(theta_1)*sin(theta_2);
%     A(6) = -0.2125*u_2*cos(theta_2);
%     A(7) = u_1*(0.11*cos(theta_1) + 0.425*cos(theta_2)*sin(theta_1)) + 0.425*u_2*cos(theta_1)*sin(theta_2);
%     A(8) = u_1*(0.11*sin(theta_1) - 0.425*cos(theta_1)*cos(theta_2)) + 0.425*u_2*sin(theta_1)*sin(theta_2);
%     A(9) = -0.425*u_2*cos(theta_2);
%     A(10) = 0.02*u_1*cos(theta_1) + 0.13075*u_1*cos(theta_2 + theta_3)*sin(theta_1) + 0.13075*u_2*sin(theta_2 + theta_3)*cos(theta_1) + 0.13075*u_3*sin(theta_2 + theta_3)*cos(theta_1) + 0.425*u_1*cos(theta_2)*sin(theta_1) + 0.425*u_2*cos(theta_1)*sin(theta_2);
%     A(11) = 0.02*u_1*sin(theta_1) - 0.13075*u_1*cos(theta_2 + theta_3)*cos(theta_1) + 0.13075*u_2*sin(theta_2 + theta_3)*sin(theta_1) + 0.13075*u_3*sin(theta_2 + theta_3)*sin(theta_1) - 0.425*u_1*cos(theta_1)*cos(theta_2) + 0.425*u_2*sin(theta_1)*sin(theta_2);
%     A(12) = - 1.0*u_2*(0.13075*cos(theta_2 + theta_3) + 0.425*cos(theta_2)) - 0.13075*u_3*cos(theta_2 + theta_3);
%     A(13) = 0.02*u_1*cos(theta_1) + 0.2615*u_1*cos(theta_2 + theta_3)*sin(theta_1) + 0.2615*u_2*sin(theta_2 + theta_3)*cos(theta_1) + 0.2615*u_3*sin(theta_2 + theta_3)*cos(theta_1) + 0.425*u_1*cos(theta_2)*sin(theta_1) + 0.425*u_2*cos(theta_1)*sin(theta_2);
%     A(14) = 0.02*u_1*sin(theta_1) - 0.2615*u_1*cos(theta_2 + theta_3)*cos(theta_1) + 0.2615*u_2*sin(theta_2 + theta_3)*sin(theta_1) + 0.2615*u_3*sin(theta_2 + theta_3)*sin(theta_1) - 0.425*u_1*cos(theta_1)*cos(theta_2) + 0.425*u_2*sin(theta_1)*sin(theta_2);
%     A(15) = - 1.0*u_2*(0.2615*cos(theta_2 + theta_3) + 0.425*cos(theta_2)) - 0.2615*u_3*cos(theta_2 + theta_3);
%     A(16) = u_1*(0.06*cos(theta_1) + 0.00025*sin(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2))) + 0.00025*u_2*cos(theta_1)*(1569.0*sin(theta_2 + theta_3) + 1700.0*sin(theta_2)) + 0.39225*u_3*sin(theta_2 + theta_3)*cos(theta_1);
%     A(17) = u_1*(0.06*sin(theta_1) - 0.00025*cos(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2))) + 0.00025*u_2*sin(theta_1)*(1569.0*sin(theta_2 + theta_3) + 1700.0*sin(theta_2)) + 0.39225*u_3*sin(theta_2 + theta_3)*sin(theta_1);
%     A(18) = - 1.0*u_2*(0.39225*cos(theta_2 + theta_3) + 0.425*cos(theta_2)) - 0.39225*u_3*cos(theta_2 + theta_3);
%     A(19) = 0.05915*u_1*cos(theta_1) + 0.0823*u_1*cos(theta_1)*cos(theta_5) + 0.425*u_1*cos(theta_2)*sin(theta_1) + 0.425*u_2*cos(theta_1)*sin(theta_2) - 0.0823*u_5*sin(theta_1)*sin(theta_5) + 0.09465*u_2*cos(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) + 0.09465*u_3*cos(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) + 0.09465*u_4*cos(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) - 0.09465*u_1*cos(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) - 0.09465*u_1*sin(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) - 0.09465*u_2*sin(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) - 0.09465*u_3*sin(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) - 0.09465*u_4*sin(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) + 0.39225*u_1*cos(theta_2)*cos(theta_3)*sin(theta_1) + 0.39225*u_2*cos(theta_1)*cos(theta_2)*sin(theta_3) + 0.39225*u_2*cos(theta_1)*cos(theta_3)*sin(theta_2) + 0.39225*u_3*cos(theta_1)*cos(theta_2)*sin(theta_3) + 0.39225*u_3*cos(theta_1)*cos(theta_3)*sin(theta_2) - 0.39225*u_1*sin(theta_1)*sin(theta_2)*sin(theta_3) - 0.0823*u_5*cos(theta_2 + theta_3 + theta_4)*cos(theta_1)*cos(theta_5) + 0.0823*u_1*cos(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.0823*u_2*sin(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) + 0.0823*u_3*sin(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) + 0.0823*u_4*sin(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5);
%     A(20) = 0.05915*u_1*sin(theta_1) - 0.425*u_1*cos(theta_1)*cos(theta_2) + 0.0823*u_1*cos(theta_5)*sin(theta_1) + 0.0823*u_5*cos(theta_1)*sin(theta_5) + 0.425*u_2*sin(theta_1)*sin(theta_2) + 0.09465*u_1*cos(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) + 0.09465*u_1*sin(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) + 0.09465*u_2*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) + 0.09465*u_3*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) + 0.09465*u_4*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) - 0.39225*u_1*cos(theta_1)*cos(theta_2)*cos(theta_3) - 0.09465*u_2*sin(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) - 0.09465*u_3*sin(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) - 0.09465*u_4*sin(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) + 0.39225*u_1*cos(theta_1)*sin(theta_2)*sin(theta_3) + 0.39225*u_2*cos(theta_2)*sin(theta_1)*sin(theta_3) + 0.39225*u_2*cos(theta_3)*sin(theta_1)*sin(theta_2) + 0.39225*u_3*cos(theta_2)*sin(theta_1)*sin(theta_3) + 0.39225*u_3*cos(theta_3)*sin(theta_1)*sin(theta_2) - 0.0823*u_1*cos(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) - 0.0823*u_5*cos(theta_2 + theta_3 + theta_4)*cos(theta_5)*sin(theta_1) + 0.0823*u_2*sin(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.0823*u_3*sin(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.0823*u_4*sin(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5);
%     A(21) = u_4*(0.09465*sin(theta_2 + theta_3 + theta_4) - 0.0823*cos(theta_2 + theta_3 + theta_4)*sin(theta_5)) - 1.0*u_3*(0.39225*cos(theta_2 + theta_3) - 0.09465*sin(theta_2 + theta_3 + theta_4) + 0.0823*cos(theta_2 + theta_3 + theta_4)*sin(theta_5)) - 1.0*u_2*(0.39225*cos(theta_2 + theta_3) + 0.425*cos(theta_2) - 0.09465*cos(theta_2 + theta_3)*sin(theta_4) - 0.09465*sin(theta_2 + theta_3)*cos(theta_4) + 0.0823*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_5) - 0.0823*sin(theta_2 + theta_3)*sin(theta_4)*sin(theta_5)) - 0.0823*u_5*sin(theta_2 + theta_3 + theta_4)*cos(theta_5);
  
end