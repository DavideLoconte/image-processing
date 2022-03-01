vidobj = getvidobj('RGB24_1280x720');
triggerconfig(vidobj, 'manual');

start(vidobj);
i = 0;

% Get a background image
img0 = getsnapshot(vidobj);

while i < 1024 % Stop after 1024 frames
    
    % Get foreground
    img = getsnapshot(vidobj);
    
    % Show the difference
    imshow(img-img0);
    drawnow;
    
    % Update image
    img0 = img;
    i = i + 1;
end

%Stop webcam
stop(vidobj);
