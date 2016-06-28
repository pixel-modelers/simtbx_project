-- MySQL Script generated by MySQL Workbench
-- Tue Jun 28 10:51:09 2016
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `mydb` DEFAULT CHARACTER SET utf8 ;
USE `mydb` ;

-- -----------------------------------------------------
-- Table `mydb`.`run`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`run` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `run` INT NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `run_UNIQUE` (`run` ASC))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`rungroup`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`rungroup` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `startrun` INT NOT NULL,
  `endrun` INT NULL,
  `active` TINYINT(1) NOT NULL,
  `config_str` TEXT NULL,
  `detector_address` VARCHAR(100) NOT NULL,
  `detz_parameter` DOUBLE NOT NULL,
  `beamx` DOUBLE NULL,
  `beamy` DOUBLE NULL,
  `binning` INT NULL,
  `energy` DOUBLE NULL,
  `untrusted_pixel_mask_path` VARCHAR(4097) NULL,
  `dark_avg_path` VARCHAR(4097) NULL,
  `dark_stddev_path` VARCHAR(4097) NULL,
  `gain_map_path` VARCHAR(4097) NULL,
  `gain_mask_level` DOUBLE NULL,
  `calib_dir` VARCHAR(4097) NULL,
  `comment` VARCHAR(1024) NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`trial`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`trial` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `trial` INT NOT NULL,
  `active` TINYINT(1) NOT NULL DEFAULT 0,
  `target_phil_str` TEXT NULL,
  `process_percent` INT NULL,
  `comment` VARCHAR(1024) NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `trial_UNIQUE` (`trial` ASC))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`job`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`job` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `status` VARCHAR(45) NULL,
  `run_id` INT NOT NULL,
  `rungroup_id` INT NOT NULL,
  `trial_id` INT NOT NULL,
  PRIMARY KEY (`id`, `run_id`, `rungroup_id`, `trial_id`),
  INDEX `fk_job_run_idx` (`run_id` ASC),
  INDEX `fk_job_rungroup1_idx` (`rungroup_id` ASC),
  INDEX `fk_job_trial1_idx` (`trial_id` ASC),
  CONSTRAINT `fk_job_run`
    FOREIGN KEY (`run_id`)
    REFERENCES `mydb`.`run` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_job_rungroup1`
    FOREIGN KEY (`rungroup_id`)
    REFERENCES `mydb`.`rungroup` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_job_trial1`
    FOREIGN KEY (`trial_id`)
    REFERENCES `mydb`.`trial` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`event`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`event` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `current_time` TIMESTAMP NOT NULL DEFAULT NOW(),
  `timestamp` VARCHAR(45) NOT NULL,
  `run_id` INT NOT NULL,
  `trial_id` INT NOT NULL,
  PRIMARY KEY (`id`, `run_id`, `trial_id`),
  INDEX `fk_event_run1_idx` (`run_id` ASC),
  INDEX `fk_event_trial1_idx` (`trial_id` ASC),
  CONSTRAINT `fk_event_run1`
    FOREIGN KEY (`run_id`)
    REFERENCES `mydb`.`run` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_event_trial1`
    FOREIGN KEY (`trial_id`)
    REFERENCES `mydb`.`trial` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`isoform`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`isoform` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(45) NOT NULL,
  `trial_id` INT NOT NULL,
  PRIMARY KEY (`id`, `trial_id`),
  INDEX `fk_isoform_trial1_idx` (`trial_id` ASC),
  CONSTRAINT `fk_isoform_trial1`
    FOREIGN KEY (`trial_id`)
    REFERENCES `mydb`.`trial` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`cell`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`cell` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `cell_a` DOUBLE NOT NULL,
  `cell_b` DOUBLE NOT NULL,
  `cell_c` DOUBLE NOT NULL,
  `cell_alpha` DOUBLE NOT NULL,
  `cell_beta` DOUBLE NOT NULL,
  `cell_gamma` DOUBLE NOT NULL,
  `lookup_symbol` VARCHAR(45) NOT NULL,
  `isoform_id` INT NULL,
  PRIMARY KEY (`id`),
  INDEX `fk_cell_isoform1_idx` (`isoform_id` ASC),
  CONSTRAINT `fk_cell_isoform1`
    FOREIGN KEY (`isoform_id`)
    REFERENCES `mydb`.`isoform` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`tag`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`tag` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(45) NOT NULL,
  `comment` VARCHAR(140) NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `name_UNIQUE` (`name` ASC))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`bin`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`bin` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `number` VARCHAR(45) NULL,
  `d_min` VARCHAR(45) NULL,
  `d_max` VARCHAR(45) NULL,
  `total_hkl` VARCHAR(45) NULL,
  `cell_id` INT NOT NULL,
  PRIMARY KEY (`id`, `cell_id`),
  INDEX `fk_bin_cell1_idx` (`cell_id` ASC),
  CONSTRAINT `fk_bin_cell1`
    FOREIGN KEY (`cell_id`)
    REFERENCES `mydb`.`cell` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`crystal`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`crystal` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `ori_1` DOUBLE NULL,
  `ori_2` DOUBLE NULL,
  `ori_3` DOUBLE NULL,
  `ori_4` DOUBLE NULL,
  `ori_5` DOUBLE NULL,
  `ori_6` DOUBLE NULL,
  `ori_7` DOUBLE NULL,
  `ori_8` DOUBLE NULL,
  `ori_9` DOUBLE NULL,
  `mosaic_block_rotation` DOUBLE NULL,
  `mosaic_block_size` DOUBLE NULL,
  `ewald_proximal_volume` DOUBLE NULL,
  `cell_id` INT NOT NULL,
  PRIMARY KEY (`id`, `cell_id`),
  INDEX `fk_crystal_cell1_idx` (`cell_id` ASC),
  CONSTRAINT `fk_crystal_cell1`
    FOREIGN KEY (`cell_id`)
    REFERENCES `mydb`.`cell` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`cell_bin`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`cell_bin` (
  `count` INT NOT NULL,
  `bin_id` INT NOT NULL,
  `crystal_id` INT NOT NULL,
  PRIMARY KEY (`bin_id`, `crystal_id`),
  INDEX `fk_cell_bin_bin1_idx` (`bin_id` ASC),
  INDEX `fk_cell_bin_crystal1_idx` (`crystal_id` ASC),
  CONSTRAINT `fk_cell_bin_bin1`
    FOREIGN KEY (`bin_id`)
    REFERENCES `mydb`.`bin` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_cell_bin_crystal1`
    FOREIGN KEY (`crystal_id`)
    REFERENCES `mydb`.`crystal` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`detector`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`detector` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `metrology` VARCHAR(4097) NULL,
  `distance` DOUBLE NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`beam`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`beam` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `direction_1` DOUBLE NOT NULL,
  `direction_2` DOUBLE NOT NULL COMMENT ' \n',
  `direction_3` DOUBLE NOT NULL,
  `wavelength` DOUBLE NOT NULL,
  `sifoil` DOUBLE NOT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`imageset`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`imageset` (
  `id` INT NOT NULL AUTO_INCREMENT,
  PRIMARY KEY (`id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`run_tag`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`run_tag` (
  `run_id` INT NOT NULL,
  `tag_id` INT NOT NULL,
  PRIMARY KEY (`run_id`, `tag_id`),
  INDEX `fk_run_tag_tag1_idx` (`tag_id` ASC),
  CONSTRAINT `fk_run_tag_run1`
    FOREIGN KEY (`run_id`)
    REFERENCES `mydb`.`run` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_run_tag_tag1`
    FOREIGN KEY (`tag_id`)
    REFERENCES `mydb`.`tag` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`imageset_event`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`imageset_event` (
  `imageset_id` INT NOT NULL,
  `event_id` INT NOT NULL,
  `event_run_id` INT NOT NULL,
  PRIMARY KEY (`imageset_id`, `event_id`, `event_run_id`),
  INDEX `fk_imageset_frame_imageset1_idx` (`imageset_id` ASC),
  INDEX `fk_imageset_frame_event1_idx` (`event_id` ASC, `event_run_id` ASC),
  CONSTRAINT `fk_imageset_frame_imageset1`
    FOREIGN KEY (`imageset_id`)
    REFERENCES `mydb`.`imageset` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_imageset_frame_event1`
    FOREIGN KEY (`event_id` , `event_run_id`)
    REFERENCES `mydb`.`event` (`id` , `run_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`experiment`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`experiment` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `beam_id` INT NOT NULL,
  `imageset_id` INT NOT NULL,
  `detector_id` INT NOT NULL,
  `crystal_id` INT NOT NULL,
  `crystal_cell_id` INT NOT NULL,
  PRIMARY KEY (`id`, `beam_id`, `imageset_id`, `detector_id`, `crystal_id`, `crystal_cell_id`),
  INDEX `fk_experiment_beam1_idx` (`beam_id` ASC),
  INDEX `fk_experiment_imageset1_idx` (`imageset_id` ASC),
  INDEX `fk_experiment_detector1_idx` (`detector_id` ASC),
  INDEX `fk_experiment_crystal1_idx` (`crystal_id` ASC, `crystal_cell_id` ASC),
  CONSTRAINT `fk_experiment_beam1`
    FOREIGN KEY (`beam_id`)
    REFERENCES `mydb`.`beam` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_experiment_imageset1`
    FOREIGN KEY (`imageset_id`)
    REFERENCES `mydb`.`imageset` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_experiment_detector1`
    FOREIGN KEY (`detector_id`)
    REFERENCES `mydb`.`detector` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_experiment_crystal1`
    FOREIGN KEY (`crystal_id` , `crystal_cell_id`)
    REFERENCES `mydb`.`crystal` (`id` , `cell_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`trial_rungroup`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`trial_rungroup` (
  `trial_id` INT NOT NULL,
  `rungroup_id` INT NOT NULL,
  PRIMARY KEY (`trial_id`, `rungroup_id`),
  INDEX `fk_trial_has_rungroup_rungroup1_idx` (`rungroup_id` ASC),
  INDEX `fk_trial_has_rungroup_trial1_idx` (`trial_id` ASC),
  CONSTRAINT `fk_trial_has_rungroup_trial1`
    FOREIGN KEY (`trial_id`)
    REFERENCES `mydb`.`trial` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_trial_has_rungroup_rungroup1`
    FOREIGN KEY (`rungroup_id`)
    REFERENCES `mydb`.`rungroup` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

