function vidobj = getvidobj(format)
    % Ottengo info sull'hardware
    hardware_info = imaqhwinfo;

    % Verifico la presenza di un dispositivo di cattura video
    % Se manca, probabilmente bisogna installare l'Add-on
    % Sul mio pc (linux64) si chiama:
    % Image acquisition toolbox support package for OS Generic Video Interface
    if (size(hardware_info.InstalledAdaptors) == 0) 
        fprintf("Missing video acquisition device\n");
        return;
    end

    device = char(cellstr(hardware_info.InstalledAdaptors(1)));
    vidobj = videoinput(device,1, format);