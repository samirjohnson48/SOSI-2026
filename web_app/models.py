from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import MetaData, String, Float, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from typing import Optional

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=convention)


# Initialize the database using the base class
db = SQLAlchemy(model_class=Base)


# Table definitions for Neon database
# Ensure that tables are defined in proper order in terms of foreign key constraints
class Asfis(db.Model):
    __tablename__ = "asfis"

    asfis_code: Mapped[str] = mapped_column(String(3), primary_key=True)
    common_name: Mapped[Optional[str]]
    family: Mapped[Optional[str]]
    isscaap_code: Mapped[Optional[int]]
    order: Mapped[Optional[str]]
    species_name: Mapped[str]
    taxonomic_code: Mapped[str] = mapped_column(String(15))


class Countries(db.Model):
    __tablename__ = "countries"

    country_un_code: Mapped[int] = mapped_column(primary_key=True)
    iso3_code: Mapped[Optional[str]] = mapped_column(String(3))
    name: Mapped[str]


class FaoAreas(db.Model):
    __tablename__ = "fao_areas"

    fao_area: Mapped[int] = mapped_column(primary_key=True)
    fao_area_name: Mapped[str] = mapped_column(unique=True)
    ocean: Mapped[Optional[str]]
    region: Mapped[Optional[str]]
    sosi_grouping: Mapped[str]


class ProductionMixin:
    uid: Mapped[str] = mapped_column(primary_key=True)
    country_un_code: Mapped[Optional[str]] = mapped_column(
        ForeignKey("countries.country_un_code"), index=True
    )
    unit: Mapped[str] = mapped_column(String(5))
    year: Mapped[int] = mapped_column(index=True)
    fao_area: Mapped[int] = mapped_column(ForeignKey("fao_areas.fao_area"), index=True)
    sosi_grouping: Mapped[Optional[str]]
    asfis_code: Mapped[str] = mapped_column(String(3), index=True)
    production: Mapped[float] = mapped_column(Float(4), default=0.0)


class Capture(db.Model, ProductionMixin):
    __tablename__ = "capture"


class Aquaculture(db.Model, ProductionMixin):
    __tablename__ = "aquaculture"
    environment_code: Mapped[str] = mapped_column(String(2))


class StockReference(db.Model):
    __tablename__ = "stock_reference"

    uid: Mapped[str] = mapped_column(primary_key=True)
    asfis_codes: Mapped[str] = mapped_column(ARRAY(String), index=True)
    fao_areas: Mapped[list[int]] = mapped_column(ARRAY(Integer), index=True)
    species_names: Mapped[str] = mapped_column(ARRAY(String))
    common_name: Mapped[str]
    sosi_grouping: Mapped[str] = mapped_column(index=True)
    isscaap_code: Mapped[int] = mapped_column(index=True)
    location: Mapped[str]
    status_year: Mapped[Optional[int]]
    tier: Mapped[Optional[int]] = mapped_column(index=True)
    status: Mapped[Optional[str]] = mapped_column(index=True)
    uncertainty: Mapped[Optional[str]]
    sosi_edition: Mapped[int]
    sosi_record_type: Mapped[str]
    ocean: Mapped[str]


class SpeciesLandings(db.Model):
    __tablename__ = "species_landings"

    uid: Mapped[str] = mapped_column(primary_key=True)
    asfis_code: Mapped[str]
    fao_area: Mapped[int] = mapped_column(ForeignKey("fao_areas.fao_area"))
    year: Mapped[int]
    production: Mapped[Optional[float]]


class StockLandings(db.Model):
    __tablename__ = "stock_landings"

    uid: Mapped[str] = mapped_column(
        ForeignKey("stock_reference.uid"), primary_key=True
    )
    landings: Mapped[float] = mapped_column(default=0.0)
