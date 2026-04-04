{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-25.05";

    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-stable,
      git-hooks,
    }:
    let
      forSystems = nixpkgs.lib.genAttrs [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      pkgsFactory =
        system: nixpkgs:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
    in
    {
      checks = forSystems (system: {
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            nixfmt.enable = true;

            gitlint.enable = true;

            clang-format.enable = true;

            ruff.enable = true;
            ruff-format.enable = true;
          };
        };
      });

      devShells = forSystems (
        system:
        let
          pkgs = pkgsFactory system nixpkgs;
          pkgsStable = pkgsFactory system nixpkgs-stable;

          inherit (self.checks.${system}.pre-commit-check) shellHook;
        in
        {
          default = pkgsStable.mkShell {
            name = "ndarray-dev-env";

            nativeBuildInputs =
              with pkgs;
              [
                cmake
                ninja
                opencode
              ]
              ++ (
                with pkgs;
                lib.optionals stdenv.cc.isClang [
                  clang-tools # needs clang-scan-deps to compile
                ]
              );

            buildInputs = with pkgsStable; [
              libtorch-bin
              armadillo
            ];

            shellHook = shellHook + ''
              export PATH=$(pwd)/build/Debug:$PATH
            '';
          };
        }
      );
    };
}
