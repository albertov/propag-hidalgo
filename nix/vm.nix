{
  # these are packages from agent-vm flake
  pkgs,
  config,
  ...
}:
let
  albertov = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDNycAdIt6d09X8drPxZVZM6gPyWl787zYY4qqdORknX2d2a/Cia4Ocp/SzF4Zb2w+iaBk/OMyjevxULFq4b2HG1CTQ5hJY//DGU8F0QVRuTdCxEGXz4x3sMEJNMA901aQbiFkOYOIMGqoSvhP65kN1y9PKMfDFatlzXZZzIDLSaxJNP6vu4jvrgWLAED58iDzaytKbQyiaTE9/anvLM1bff9uIEHjcA1OW4/5dm85lmh0B6p34FLfXeDaqo+FNSymhniKqu5zulKonz5emgwpnWzFahpFh6hirP5RncS6dH+q1wubDoMY2Dxk2QW4TMx0iDSMpwbgpq78h1p6t4gL5AINehzUOBVmNkn9J/slJ5RCyjiHQdN9uNBambeklJCZbeVLW0ZirMIgj1UyKpm+LDRedWc4Kifz/B5VoxBewf06Pn8JPfvzLVlcB201+3NWpIwRgp/DsVHPsaIGeh7VGH9Et965ssTYCxytS2uFixGXpT1gcM0+KQkfw3v0svyw5UOmr9prCgT1Zyl6DJ68b4puH90pV9VThH07JCdJrBV/Y6r7yilCpswO2ADawlMp5/ZU3/pQ+7BIISiIA1ddmADnGRmKHsSXV4n8xGleR3szmLxtjyCKhY9znhrjlc4VLryBPjqyzOtR4PVmWBscchmP8dyPCZlx2V3qYayzymQ== Alberto Valverde Gonzalez";
in
{
  # Configure host nix store as binary cache
  nix.settings.substituters = [
    "http://10.0.2.2:5000"
  ];
  nix.settings.trusted-public-keys = [
    "alberto-valverde-1:A+NbXRfx+Uo0tQNZ8hlip+1zru2P32l7/skPDeaZnxU="
  ];
  # For Claude Code
  services.openssh.enable = true;
  services.openssh.settings.PasswordAuthentication = false;
  users.extraUsers.mcp-proxy.openssh.authorizedKeys.keys = [ albertov ];
  virtualisation.forwardPorts = [
    {
      from = "host";
      host.port = 6222;
      guest.port = 22;
    }
  ];
  # So we can run uvx code
  programs.nix-ld.enable = true;
  programs.bash = {
    shellAliases = {
      # WARNING: CC in yolo mode, make sure to push checkout to a safe
      # place before letting CC go wild and don't leave any secret
      # inside the /workspace (except the inevitable claude auth token)
      claude = "${pkgs.nodejs}/bin/npx @anthropic-ai/claude-code --dangerously-skip-permissions";
    };
  };
  environment.systemPackages = with pkgs; [
    # So we can easily add it as a stdio server for claude code
    mcp-language-server
    nodejs
    uv
    jq
    ripgrep
    fd
    bat
    eza
    vim
  ];
  programs.tmux = {
    enable = true;
    clock24 = true;
    extraConfig = ''
      set-option -g prefix C-a
      set-option -g default-shell /run/current-system/sw/bin/bash
      bind-key C-a last-window
    '';
  };
  # We have CC already
  services.mcp-proxy.namedServers.codemcp.enabled = false;
  services.mcp-proxy.namedServers = {
    nixos = {
      enabled = true;
      runtimeInputs = with pkgs; [
        nix
        git
      ];
      # TODO: Just add this flake as an input, we don't like
      # stuff randomly updating while sipping our morning coffee/tea
      command = "nix";
      args = [
        "run"
        "github:utensils/mcp-nixos"
        "--"
      ];
    };
  };
}
