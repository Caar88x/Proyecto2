# ===========================================================================
# PROYECTO: FotoApp - Procesador de imágenes para redes sociales
# ===========================================================================


# ===========================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ===========================================================================

from abc import ABC, abstractmethod
# QUÉ SIGNIFICA:
#   "from"      -> palabra clave que inicia un import selectivo.
#   "abc"       -> módulo estándar "Abstract Base Classes".
#   "import"    -> palabra clave que introduce los nombres a importar.
#   "ABC"       -> metaclase base para clases abstractas.
#   ","         -> separador entre nombres a importar.
#   "abstractmethod" -> decorador que marca métodos obligatorios.
# PARA QUÉ:
#   - Definir la interfaz abstracta ImageProcessor con un método obligatorio.
#   - Garantizar que TODAS las subclases implementen process().
# POR QUÉ:
#   - ABC impide instanciar la base; abstractmethod obliga a implementar.
#   - Importar solo lo necesario (no "import abc" completo) -> namespace limpio.
# QUÉ HACE:
#   - Deja disponibles ABC y abstractmethod en este módulo.
# SE UTILIZA PARA:
#   Se utiliza para declarar que ImageProcessor es una clase abstracta y para
#   marcar process() como un método que todas las subclases están obligadas a
#   implementar. Sin estos dos elementos, el contrato no existiría y cualquier
#   procesador podría olvidar implementar process() sin que Python lo detecte.
# SI SE QUITA:
#   - No podrías definir la interfaz ImageProcessor.
#   - Subclases podrían olvidar process() y fallar tardíamente con AttributeError.
# TRAMPA: "¿ABC es metaclase?" -> Sí, usa ABCMeta internamente.

from io import BytesIO
# QUÉ SIGNIFICA:
#   "from"    -> import selectivo.
#   "io"      -> módulo estándar de I/O.
#   "import"  -> introduce nombres.
#   "BytesIO" -> clase que simula archivo binario en RAM.
# PARA QUÉ:
#   - Envolver bytes descargados en un file-like para Image.open.
# POR QUÉ:
#   - Pillow acepta file-like. BytesIO evita escribir en disco (más rápido,
#     sin archivos temporales, sin problemas de permisos).
# QUÉ HACE:
#   - Crea un archivo virtual en RAM a partir de bytes.
# SE UTILIZA PARA:
#   Se utiliza para tomar los bytes crudos que devuelve requests (resp.content)
#   y convertirlos en un objeto que Image.open pueda leer, sin necesidad de
#   crear un archivo temporal en el disco. Así la descarga y la apertura de la
#   imagen ocurren completamente en memoria RAM.
# SI SE QUITA:
#   - Habría que guardar en disco o pasar bytes (Image.open no acepta bytes crudos).
# TRAMPA: "¿BytesIO vs StringIO?" -> BytesIO bytes; StringIO texto.

from dataclasses import dataclass
# QUÉ SIGNIFICA:
#   "from"        -> import selectivo.
#   "dataclasses" -> módulo estándar.
#   "import"      -> introduce nombres.
#   "dataclass"   -> decorador que genera __init__/__repr__/__eq__.
# PARA QUÉ:
#   - Definir ProcessResult como contenedor tipado con 5 campos.
# POR QUÉ:
#   - Genera boilerplate automáticamente; tipado fuerte; documenta el contrato.
# QUÉ HACE:
#   - Al aplicarlo a ProcessResult, genera __init__/__repr__/__eq__.
# SE UTILIZA PARA:
#   Se utiliza para decorar la clase ProcessResult y que Python genere
#   automáticamente el constructor, la representación legible y la comparación.
#   Así no hay que escribir manualmente __init__, __repr__ ni __eq__, y el
#   contrato de retorno de process() queda documentado en el código.
# SI SE QUITA:
#   - Clase manual con __init__/__repr__/__eq__ (más código, más errores).
# TRAMPA: "¿Por qué no NamedTuple?" -> dataclass permite métodos y defaults.

from typing import Callable, Optional, Dict, Any, List
# QUÉ SIGNIFICA:
#   "from"     -> import selectivo.
#   "typing"   -> módulo de type hints.
#   "import"   -> introduce nombres.
#   "Callable" -> tipo para invocables (funciones/clases).
#   ","        -> separador.
#   "Optional" -> T | None.
#   "Dict"     -> diccionario tipado (Dict[K, V]).
#   "Any"      -> cualquier tipo (usado en info()).
#   "List"     -> lista tipada (List[T]).
# PARA QUÉ:
#   - Callable[[str], ImageProcessor]: tipar el factory de filtros.
#   - Optional[Image.Image]: tipar self.img (puede ser None).
#   - Dict[str, Any]: tipar el dict de info().
#   - List[str]: tipar listas (filtros válidos, etc.).
# POR QUÉ:
#   - Callable es más preciso que Type para factories con args.
#   - Optional documenta que un valor puede ser None (mypy obliga a validar).
#   - Dict/List/Any permiten tipar funciones nuevas (info, helpers).
# QUÉ HACE:
#   - Anotaciones de tipo (no afectan en runtime; solo documentación/validación).
# SE UTILIZA PARA:
#   Se utiliza para anotar dos cosas: el factory de filtros (que recibe un
#   string y devuelve un ImageProcessor) y el estado de la imagen actual en
#   FotoApp (que puede ser None si todavía no se ha cargado nada). Además,
#   Dict/List/Any tipan las funciones nuevas (info(), _safe_filename()).
#   Estas anotaciones permiten que mypy o el IDE detecten errores antes de
#   ejecutar.
# SI SE QUITA:
#   - El código funciona; pierdes validación estática y documentación.
# TRAMPA: "¿Callable vs Type?" -> Callable acepta cualquier invocable; Type exige clase.

import cv2, requests, numpy as np
# QUÉ SIGNIFICA:
#   "import"   -> palabra clave.
#   "cv2"      -> OpenCV (visión por computador).
#   ","        -> separador entre módulos.
#   "requests" -> HTTP de alto nivel.
#   ","        -> separador.
#   "numpy"    -> arreglos N-D.
#   "as"       -> palabra clave para alias.
#   "np"       -> alias estándar (PEP 8 permite alias conocidos).
# PARA QUÉ:
#   - cv2: cvtColor, equalizeHist, Canny, Laplacian (procesamiento).
#   - requests: get con timeout y raise_for_status (descarga).
#   - np: np.array(img) -> ndarray (puente PIL <-> OpenCV).
# POR QUÉ:
#   - cv2: algoritmos que Pillow no tiene (Canny, YCrCb).
#   - requests: más legible que urllib; maneja timeouts/redirects/SSL.
#   - np: conversión necesaria entre PIL y OpenCV.
# QUÉ HACE:
#   - Deja disponibles los 3 módulos (con alias np para numpy).
# SE UTILIZA PARA:
#   Se utiliza para tener a mano las tres herramientas centrales del proyecto:
#   OpenCV para las operaciones de visión (conversión de color, ecualización,
#   detección de bordes), requests para descargar imágenes desde internet y
#   NumPy para convertir entre el formato de PIL y el de OpenCV.
# SI SE QUITA:
#   - No podrías procesar imágenes ni descargarlas.
# TRAMPA: "¿OpenCV usa SIMD siempre?" -> No; C++ optimizado, SIMD según build/CPU.
# TRAMPA: "¿requests es síncrono?" -> Sí; para concurrencia usa aiohttp/ThreadPool.

import matplotlib.pyplot as plt
# QUÉ SIGNIFICA:
#   "import"            -> palabra clave.
#   "matplotlib.pyplot" -> módulo de gráficos.
#   "as"                -> alias.
#   "plt"               -> alias estándar.
# PARA QUÉ:
#   - Comparar original vs procesada en 1x2 y guardar PNG.
# POR QUÉ:
#   - cv2.imshow requiere GUI nativa (falla en Colab/headless).
#   - PIL.Image.show solo muestra una imagen (sin comparación).
#   - matplotlib permite layout comparativo + savefig.
# QUÉ HACE:
#   - Deja disponible plt para subplots/imshow/savefig/show/close.
# SE UTILIZA PARA:
#   Se utiliza para crear una figura con dos paneles (original y procesada),
#   dibujar cada imagen con su título, guardar la comparación como archivo PNG
#   en el disco y mostrar la ventana al usuario. Es la herramienta que produce
#   toda la salida visual del programa.
# SI SE QUITA:
#   - No podrías comparar ni guardar PNGs desde matplotlib.
# TRAMPA: "¿plt.show() bloquea siempre?" -> No; solo en backends interactivos.

from PIL import Image, ImageFilter
# QUÉ SIGNIFICA:
#   "from"         -> import selectivo.
#   "PIL"          -> Pillow (fork de PIL).
#   "import"       -> introduce nombres.
#   "Image"        -> clase principal de imagen.
#   ","            -> separador.
#   "ImageFilter"  -> filtros predefinidos.
# PARA QUÉ:
#   - Abrir, convertir a RGB, redimensionar con LANCZOS, aplicar filtros.
# POR QUÉ:
#   - Pillow es el estándar de I/O de imágenes en Python.
#   - Filtros de alto nivel limpios (ImageFilter).
# QUÉ HACE:
#   - Deja disponibles Image e ImageFilter.
# SE UTILIZA PARA:
#   Se utiliza Image para abrir archivos de imagen, convertir entre modos de
#   color, redimensionar con LANCZOS y guardar resultados. Se utiliza
#   ImageFilter para tener disponibles las constantes de los nueve filtros
#   espaciales (BLUR, SHARPEN, EMBOSS, etc.) que el usuario puede elegir.
# SI SE QUITA:
#   - No podrías abrir/guardar imágenes ni aplicar filtros.
# TRAMPA: "¿PIL y Pillow?" -> PIL abandonado (2011); Pillow es fork mantenido.

import os
# QUÉ SIGNIFICA: "import" + "os" (módulo estándar de sistema operativo).
# PARA QUÉ: comprobar existencia de rutas locales en load_local().
# POR QUÉ: os.path.exists es la forma estándar de verificar archivos.
# QUÉ HACE: expone os.path.exists, os.path.isfile, os.makedirs.
# SE UTILIZA PARA: Se utiliza en ImageLoader.load_local() para validar que la
#   ruta exista antes de intentar Image.open (mejor mensaje de error), y en
#   ImageLoader.save() para crear el directorio destino si no existe.
# SI SE QUITA: los errores de ruta serían FileNotFoundError crudos de Pillow.
# TRAMPA: "¿os vs pathlib?" -> pathlib es más moderno; os es más simple aquí.

import re
# QUÉ SIGNIFICA: "import" + "re" (módulo de expresiones regulares).
# PARA QUÉ: sanitizar nombres de archivo en _safe_filename().
# POR QUÉ: nombres con "/", "\", ":" o espacios rompen rutas en distintos SO.
# QUÉ HACE: expone re.sub para reemplazar caracteres no seguros.
# SE UTILIZA PARA: Se utiliza en el helper _safe_filename() para convertir
#   "edge enhance" -> "edge_enhance" y eliminar caracteres problemáticos,
#   garantizando que savefig() escriba un nombre portátil.
# SI SE QUITA: nombres con espacios o ":" fallarían en Windows.
# TRAMPA: "¿Por qué no .replace(' ','_')?" -> replace solo cubre espacios;
#   re.sub cubre cualquier carácter no alfanumérico.


# ===========================================================================
# 2. CONSTANTES GLOBALES
# ===========================================================================

ENLACES = {
# QUÉ SIGNIFICA:
#   "ENLACES" -> nombre de constante (UPPER_SNAKE_CASE por PEP 8).
#   "="       -> operador de asignación.
#   "{"       -> apertura de diccionario literal.
# PARA QUÉ:
#   - Mapear claves cortas ("instagram") a URLs de prueba reproducibles.
# POR QUÉ:
#   - Evitar teclear URLs largas en demos.
#   - picsum.photos con seed garantiza la MISMA imagen para la MISMA URL.
# QUÉ HACE:
#   - Define un diccionario clave -> URL.
# SE UTILIZA PARA:
#   Se utiliza para que el usuario pueda escribir una clave corta como
#   "instagram" en lugar de una URL completa, y para tener un conjunto de
#   imágenes de prueba reproducibles durante las demostraciones. ImageLoader
#   consulta este diccionario para traducir la clave a la URL real.
# SI SE QUITA:
#   - El usuario tendría que pegar URLs completas.
# TRAMPA: "¿Y si picsum cae?" -> ConnectionError capturado por except Exception.
    "instagram": "https://picsum.photos/seed/insta/1200/1200",
    "facebook":  "https://picsum.photos/seed/face/1200/630",
    "twitter":   "https://picsum.photos/seed/twit/1200/675",
    "youtube":   "https://picsum.photos/seed/yt/1280/720",
    "paisaje":   "https://picsum.photos/seed/paisaje/1600/900",
    "retrato":   "https://picsum.photos/seed/retrato/800/800",
}

DIMENSIONES_REDES = {
# QUÉ SIGNIFICA:
#   "DIMENSIONES_REDES" -> constante.
#   "="                 -> asignación.
#   "{"                 -> apertura de dict.
# PARA QUÉ:
#   - Definir el tamaño EXACTO objetivo de cada red social.
# POR QUÉ:
#   - _ajustar_a_red usa estas dimensiones para componer el lienzo final.
#   - Tupla = inmutable, hashable, semánticamente (w, h).
# QUÉ HACE:
#   - Mapea red -> tupla (ancho, alto) EXACTA.
# SE UTILIZA PARA:
#   Se utiliza para que la función _ajustar_a_red sepa exactamente qué ancho y
#   alto debe tener el lienzo final de cada red social. Sin este diccionario,
#   el programa solo podría limitar el tamaño máximo, pero no garantizar las
#   dimensiones exactas que cada red exige.
# SI SE QUITA:
#   - thumbnail solo limitaría máximos (no exactos).
# TRAMPA: "¿Y si teclea 'Instagram'?" -> red.lower() normaliza.
    "instagram": (1080, 1080),
    "facebook":  (1200, 630),
    "twitter":   (1200, 675),
    "youtube":   (1280, 720),
}


def _safe_filename(name: str) -> str:
    # QUÉ SIGNIFICA:
    #   "def"            -> define función.
    #   "_safe_filename" -> nombre (prefijo "_" = privado del módulo).
    #   "("              -> apertura.
    #   "name"           -> parámetro.
    #   ":"              -> separador.
    #   "str"            -> tipo.
    #   ")"              -> cierre.
    #   "->"             -> retorno.
    #   "str"            -> tipo de retorno.
    #   ":"              -> bloque.
    # PARA QUÉ: convertir un nombre arbitrario en un nombre de archivo seguro.
    # POR QUÉ: nombres con espacios, "/", ":", "*" rompen rutas en Windows/macOS.
    # QUÉ HACE: reemplaza cualquier carácter no [A-Za-z0-9_-] por "_".
    # SE UTILIZA PARA: Se utiliza en FilterProcessor.process() para construir
    #   "f_edge_enhance.png" y en ImageLoader.save() para sanitizar rutas.
    # SI SE QUITA: nombres con espacios o caracteres especiales fallarían.
    # TRAMPA: "¿Por qué no usar solo .replace(' ', '_')?" -> replace solo
    #   cubre espacios; aquí cubrimos CUALQUIER carácter problemático.
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "archivo"


# ===========================================================================
# 3. CONTRATO TIPADO
# ===========================================================================

@dataclass(frozen=True)
# QUÉ SIGNIFICA:
#   "@"           -> sintaxis de decorador.
#   "dataclass"   -> decorador importado.
#   "("           -> apertura de argumentos.
#   "frozen"      -> nombre del parámetro.
#   "="           -> asignación de argumento keyword.
#   "True"        -> valor booleano.
#   ")"           -> cierre de argumentos.
# PARA QUÉ:
#   - Convertir ProcessResult en dataclass inmutable con __init__/__repr__/__eq__.
# POR QUÉ:
#   - frozen=True: inmutabilidad + hashabilidad (seguro para concurrencia).
#   - Evita mutaciones accidentales del resultado (defensivo).
# QUÉ HACE:
#   - Genera boilerplate y bloquea asignaciones post-init.
# SE UTILIZA PARA:
#   Se utiliza para que ProcessResult sea una clase de datos inmutable: que el
#   constructor, la representación y la comparación se generen solos, y que
#   nadie pueda modificar los campos después de crear la instancia. Esto evita
#   que un procesador altere accidentalmente el resultado de otro.
# SI SE QUITA:
#   - __init__/__repr__/__eq__ manuales; mutable (riesgo de bugs).
# TRAMPA: "¿Por qué frozen=True?" -> Evita mutaciones; permite usar como clave de dict.
class ProcessResult:
    # QUÉ SIGNIFICA: "class" definición; "ProcessResult" nombre (CamelCase); ":" bloque.
    # PARA QUÉ: contrato de retorno de process().
    # POR QUÉ: nombre descriptivo del propósito.
    # QUÉ HACE: define la estructura de datos.
    # SE UTILIZA PARA: Se utiliza como tipo de retorno de todos los procesadores
    #   y como entrada de ImageVisualizer.compare. Es el "paquete" que viaja
    #   desde el procesador hasta el visualizador.
    # SI SE QUITA: no habría contrato de retorno.
    """Resultado tipado de un procesamiento de imagen."""
    # QUÉ SIGNIFICA: '"""..."""' docstring.
    # PARA QUÉ: documentar la clase para help() e IDEs.
    # POR QUÉ: convención PEP 257.
    # QUÉ HACE: describe la clase.
    # SE UTILIZA PARA: Se utiliza para que help(ProcessResult) muestre la
    #   descripción y para que los IDEs muestren el tooltip.
    # SI SE QUITA: sin documentación (no afecta ejecución).

    original: Image.Image
    # QUÉ SIGNIFICA: "original" nombre; ":" separador; "Image.Image" tipo.
    # PARA QUÉ: imagen izquierda (sin procesar).
    # POR QUÉ: compare() necesita ambas imágenes para comparar.
    # QUÉ HACE: declara el campo 0 del dataclass.
    # SE UTILIZA PARA: Se utiliza para que ImageVisualizer.compare dibuje la
    #   imagen original en el panel izquierdo (ax[0]) con su título.
    # SI SE QUITA: compare() no tendría imagen izquierda.
    procesada: Image.Image
    # QUÉ SIGNIFICA: campo de tipo PIL.Image.
    # PARA QUÉ: imagen derecha (procesada).
    # POR QUÉ: es el resultado del algoritmo aplicado.
    # QUÉ HACE: declara el campo 1.
    # SE UTILIZA PARA: Se utiliza para que ImageVisualizer.compare dibuje la
    #   imagen procesada en el panel derecho (ax[1]) con su título.
    # SI SE QUITA: compare() no tendría imagen derecha.
    titulo_izq: str
    # QUÉ SIGNIFICA: campo de tipo str.
    # PARA QUÉ: título del panel izquierdo.
    # POR QUÉ: el usuario debe saber qué está viendo a la izquierda.
    # QUÉ HACE: declara el campo 2.
    # SE UTILIZA PARA: Se utiliza para que ax[0].set_title muestre "Original"
    #   encima del panel izquierdo de la comparación.
    # SI SE QUITA: compare() no tendría título izquierdo.
    titulo_der: str
    # QUÉ SIGNIFICA: campo de tipo str.
    # PARA QUÉ: título del panel derecho.
    # POR QUÉ: el usuario debe saber qué procesamiento se aplicó.
    # QUÉ HACE: declara el campo 3.
    # SE UTILIZA PARA: Se utiliza para que ax[1].set_title muestre "Contraste",
    #   "Blur", "Canny", etc., encima del panel derecho de la comparación.
    # SI SE QUITA: compare() no tendría título derecho.
    archivo: str
    # QUÉ SIGNIFICA: campo de tipo str.
    # PARA QUÉ: nombre del PNG de salida.
    # POR QUÉ: compare() debe guardar la comparación en disco.
    # QUÉ HACE: declara el campo 4.
    # SE UTILIZA PARA: Se utiliza para que plt.savefig(result.archivo) escriba
    #   el archivo PNG en disco con ese nombre ("c.png", "f_blur.png", etc.).
    # SI SE QUITA: compare() no sabría dónde guardar.


class ImageProcessor(ABC):
    # QUÉ SIGNIFICA: "class" definición; "ImageProcessor" nombre; "(ABC)" herencia; ":" bloque.
    # PARA QUÉ: contrato abstracto para todos los procesadores.
    # POR QUÉ: ABC impide instanciar la base; obliga a implementar process().
    # QUÉ HACE: define el contrato.
    # SE UTILIZA PARA: Se utiliza como tipo de las dependencias que FotoApp
    #   recibe en su constructor (contrast_processor, filter_factory,
    #   edge_processor, boceto_processor). Gracias a esto, FotoApp no depende
    #   de clases concretas sino de esta abstracción (DIP).
    # SI SE QUITA: no habría contrato; cada procesador sería independiente.
    # TRAMPA: "¿Por qué ABC y no Protocol?" -> ABC fuerza error en runtime.
    """Interfaz abstracta para procesadores. Cumple OCP y DIP."""

    @abstractmethod
    # QUÉ SIGNIFICA: "@" decorador; "abstractmethod" nombre.
    # PARA QUÉ: marcar process() como obligatorio.
    # POR QUÉ: sin esto, subclases podrían no implementarlo (fallo tardío).
    # QUÉ HACE: marca el método como abstracto.
    # SE UTILIZA PARA: Se utiliza para obligar a ContrastProcessor,
    #   FilterProcessor, EdgeDetectionProcessor y BocetoProcessor a implementar
    #   process(). Si alguna no lo hace, Python lanza TypeError al instanciarla.
    # SI SE QUITA: la subclase no estaría obligada a implementar process().
    def process(self, img: Image.Image) -> ProcessResult:
        # QUÉ SIGNIFICA:
        #   "def"           -> define función.
        #   "process"       -> nombre del método.
        #   "("             -> apertura.
        #   "self"          -> referencia a instancia.
        #   ","             -> separador.
        #   "img"           -> parámetro.
        #   ":"             -> separador.
        #   "Image.Image"   -> tipo.
        #   ")"             -> cierre.
        #   "->"            -> retorno.
        #   "ProcessResult" -> tipo de retorno.
        #   ":"             -> bloque.
        # PARA QUÉ: contrato: recibe Image, devuelve ProcessResult.
        # POR QUÉ: tipado fuerte permite mypy validar acceso a campos.
        # QUÉ HACE: declara el método sin implementarlo.
        # SE UTILIZA PARA: Se utiliza como firma común que FotoApp invoca en
        #   las opciones 2, 3, 4 y 5 del menú.
        # SI SE QUITA: no habría contrato; FotoApp no sabría qué esperar.
        # TRAMPA: "¿Por qué ProcessResult y no tuple?" -> tuple no documenta campos.
        pass
        # QUÉ SIGNIFICA: "pass" = no-op (cuerpo vacío obligatorio).
        # PARA QUÉ: cumplir sintaxis (bloque no puede estar vacío).
        # POR QUÉ: Python requiere al menos una sentencia en un bloque.
        # QUÉ HACE: no hace nada.
        # SE UTILIZA PARA: Se utiliza para que el método abstracto tenga un
        #   cuerpo válido sin implementar lógica; la implementación real la
        #   aportan las subclases.
        # SI SE QUITA: SyntaxError.


# ===========================================================================
# 4. ImageLoader
# ===========================================================================

class ImageLoader:
    # QUÉ SIGNIFICA: "class" + "ImageLoader" + ":".
    # PARA QUÉ: encapsular la carga de imágenes (SRP).
    # POR QUÉ: si mañana se carga desde S3/GCS, SOLO cambia esta clase (OCP).
    # QUÉ HACE: define la clase.
    # SE UTILIZA PARA: Se utiliza en la opción 1 del menú para convertir una
    #   clave, URL o ruta local en un objeto PIL.Image listo para procesar.
    # SI SE QUITA: la lógica de carga se mezclaría con el controlador.
    """Carga y resolución de fuentes de imágenes. [SRP]"""

    @staticmethod
    # QUÉ SIGNIFICA: "@" decorador; "staticmethod" nombre.
    # PARA QUÉ: load() no recibe self.
    # POR QUÉ: no usa estado de instancia; es utilidad pura.
    # QUÉ HACE: convierte load() en método estático.
    # SE UTILIZA PARA: Se utiliza para poder llamar ImageLoader.load(k) sin
    #   necesidad de crear una instancia de ImageLoader. Es una utilidad pura.
    # SI SE QUITA: tendrías que crear ImageLoader() solo para llamar load() (antipatrón).
    # TRAMPA: "¿Por qué no @classmethod?" -> classmethod recibe cls; aquí no se necesita.
    def load(source: str) -> Image.Image:
        # QUÉ SIGNIFICA:
        #   "def"         -> define.
        #   "load"        -> nombre.
        #   "("           -> apertura.
        #   "source"      -> parámetro.
        #   ":"           -> separador.
        #   "str"         -> tipo.
        #   ")"           -> cierre.
        #   "->"          -> retorno.
        #   "Image.Image" -> tipo.
        #   ":"           -> bloque.
        # PARA QUÉ: cargar desde clave/URL/ruta; devolver PIL RGB.
        # POR QUÉ: un solo tipo de parámetro unifica 3 formatos.
        # QUÉ HACE: resuelve, descarga, abre, normaliza a RGB.
        # SE UTILIZA PARA: Se utiliza en FotoApp.menu opción 1 (a través de
        #   redimensionar_para_red) y en cargar_imagen().
        # SI SE QUITA: no podrías cargar imágenes.
        # TRAMPA: "¿Qué pasa si source es None?" -> .lower() lanza AttributeError.

        url = ENLACES.get(source.lower(), source)
        # QUÉ SIGNIFICA:
        #   "url"            -> variable.
        #   "="              -> asignación.
        #   "ENLACES"        -> dict.
        #   "."              -> acceso.
        #   "get"            -> método.
        #   "("              -> apertura.
        #   "source"         -> argumento.
        #   "."              -> acceso.
        #   "lower"          -> método.
        #   "()"             -> llamada.
        #   ","              -> separador.
        #   "source"         -> default.
        #   ")"              -> cierre.
        # PARA QUÉ: resolver fuente: clave -> URL, o dejar URL/ruta.
        # POR QUÉ: .get() evita KeyError; .lower() normaliza.
        # QUÉ HACE: "instagram" -> URL; "https://..." -> sin cambios; "/ruta" -> sin cambios.
        # SE UTILIZA PARA: Se utiliza para traducir lo que el usuario escribió
        #   ("instagram", una URL o una ruta) en una URL definitiva que se
        #   descargará o abrirá en los pasos siguientes.
        # SI SE QUITA: if/else manual (más verboso).
        # TRAMPA: "¿Y si source es mayúscula?" -> .lower() lo normaliza.

        if url.startswith("http"):
            # QUÉ SIGNIFICA:
            #   "if"         -> condicional.
            #   "url"        -> variable.
            #   "."          -> acceso.
            #   "startswith" -> método.
            #   "("          -> apertura.
            #   ""http""     -> string.
            #   ")"          -> cierre.
            #   ":"          -> bloque.
            # PARA QUÉ: distinguir URL web de ruta local.
            # POR QUÉ: "http" matchea http:// y https://.
            # QUÉ HACE: True si es URL web.
            # SE UTILIZA PARA: Se utiliza para decidir si hay que descargar la
            #   imagen (rama URL) o si hay que abrirla directamente del disco
            #   (rama else). Es la bifurcación principal de load().
            # SI SE QUITA: no podrías distinguir URL de ruta.
            resp = requests.get(url, timeout=15)
            # QUÉ SIGNIFICA:
            #   "resp"     -> variable (Response).
            #   "="        -> asignación.
            #   "requests" -> módulo.
            #   "."        -> acceso.
            #   "get"      -> método HTTP.
            #   "("        -> apertura.
            #   "url"      -> argumento.
            #   ","        -> separador.
            #   "timeout"  -> keyword.
            #   "="        -> asignación.
            #   "15"       -> valor.
            #   ")"        -> cierre.
            # PARA QUÉ: descargar con timeout de 15 s.
            # POR QUÉ: timeout evita bloqueo indefinido.
            # QUÉ HACE: hace GET HTTP, devuelve Response.
            # SE UTILIZA PARA: Se utiliza para descargar los bytes de la imagen
            #   desde la URL resuelta. El timeout de 15 segundos garantiza que
            #   si el servidor no responde, el programa falle en lugar de
            #   quedarse colgado.
            # SI SE QUITA: el programa podría colgarse.
            # TRAMPA: "¿Por qué 15 s?" -> Empírico; razonable para <10 MB.
            resp.raise_for_status()
            # QUÉ SIGNIFICA: "resp" + "." + "raise_for_status" + "()".
            # PARA QUÉ: lanzar HTTPError si 4xx/5xx.
            # POR QUÉ: requests NO lanza por defecto; esto sí (fail-fast).
            # QUÉ HACE: lanza HTTPError si status_code >= 400.
            # SE UTILIZA PARA: Se utiliza para detener la ejecución con un error
            #   claro si el servidor devolvió 404 o 500, en lugar de intentar
            #   abrir un HTML de error como si fuera una imagen.
            # SI SE QUITA: un 404 pasaría silencioso; Image.open fallaría confuso.
            # TRAMPA: "¿raise_for_status()?" -> Sin él, 404 devuelve HTML.
            content = BytesIO(resp.content)
            # QUÉ SIGNIFICA:
            #   "content"     -> variable (BytesIO).
            #   "="           -> asignación.
            #   "BytesIO"     -> clase.
            #   "("           -> apertura.
            #   "resp"        -> Response.
            #   "."           -> acceso.
            #   "content"     -> atributo bytes.
            #   ")"           -> cierre.
            # PARA QUÉ: envolver bytes en file-like para Image.open.
            # POR QUÉ: .content (no .text) porque imágenes son binarias.
            # QUÉ HACE: crea archivo virtual en RAM.
            # SE UTILIZA PARA: Se utiliza para convertir los bytes descargados
            #   en un archivo virtual que Image.open pueda leer, sin escribir
            #   nada en el disco.
            # SI SE QUITA: Image.open no aceptaría bytes directamente.
            # TRAMPA: "¿.content vs .text?" -> .text corrompería bytes.
        else:
            # QUÉ SIGNIFICA: "else" + ":".
            # PARA QUÉ: rama ruta local.
            # POR QUÉ: no todas las fuentes son URLs.
            # QUÉ HACE: ejecuta el bloque alternativo.
            # SE UTILIZA PARA: Se utiliza para el caso en que source es una
            #   ruta local (por ejemplo, "/home/user/foto.jpg"). En ese caso
            #   no hay que descargar nada, solo pasar la ruta a Image.open.
            # SI SE QUITA: no podrías cargar locales.
            content = url
            # QUÉ SIGNIFICA: "content" + "=" + "url".
            # PARA QUÉ: pasar ruta tal cual.
            # POR QUÉ: Image.open acepta str directamente.
            # QUÉ HACE: asigna la ruta a content.
            # SE UTILIZA PARA: Se utiliza para que el paso siguiente (Image.open)
            #   reciba una ruta de archivo local y abra la imagen del disco.
            # SI SE QUITA: content quedaría indefinido.

        img = Image.open(content)
        # QUÉ SIGNIFICA:
        #   "img"       -> variable (PIL.Image).
        #   "="         -> asignación.
        #   "Image"     -> módulo.
        #   "."         -> acceso.
        #   "open"      -> función.
        #   "("         -> apertura.
        #   "content"   -> BytesIO o str.
        #   ")"         -> cierre.
        # PARA QUÉ: abrir imagen (detecta formato automáticamente).
        # POR QUÉ: lazy; no carga píxeles hasta necesitarlos.
        # QUÉ HACE: abre la imagen en memoria.
        # SE UTILIZA PARA: Se utiliza para obtener un objeto PIL.Image a partir
        #   del BytesIO (si vino de internet) o de la ruta (si vino del disco).
        # SI SE QUITA: no habría imagen.
        # TRAMPA: "¿Se cierra?" -> Lazy; GC libera.

        return ImageLoader._normalizar(img)
        # QUÉ SIGNIFICA: "return" + "ImageLoader._normalizar(img)".
        # PARA QUÉ: delegar la conversión RGBA/RGB al helper (DRY).
        # POR QUÉ: load(), load_from_bytes() y load_local() comparten esta lógica.
        # QUÉ HACE: devuelve la imagen normalizada a RGB.
        # SE UTILIZA PARA: Se utiliza para garantizar que todos los
        #   procesadores reciban PIL.Image en modo RGB.
        # SI SE QUITA: duplicarías el bloque de transparencia.

    @staticmethod
    # QUÉ SIGNIFICA: "@" + "staticmethod".
    # PARA QUÉ: load_from_bytes() no recibe self.
    # POR QUÉ: utilidad pura, no usa estado de instancia.
    # QUÉ HACE: convierte load_from_bytes() en método estático.
    # SE UTILIZA PARA: Se utiliza en notebooks y tests para cargar una imagen
    #   directamente desde bytes (por ejemplo, leídos de un archivo o de una
    #   respuesta HTTP guardada en caché), sin pasar por la red.
    # SI SE QUITA: tendrías que crear BytesIO manualmente en cada test.
    # TRAMPA: "¿load_from_bytes vs load?" -> load() resuelve claves/URLs;
    #   load_from_bytes() solo decodifica bytes (más explícito en tests).
    def load_from_bytes(data: bytes) -> Image.Image:
        # QUÉ SIGNIFICA:
        #   "def"             -> define.
        #   "load_from_bytes" -> nombre descriptivo.
        #   "("               -> apertura.
        #   "data"            -> parámetro (bytes).
        #   ":"               -> separador.
        #   "bytes"           -> tipo.
        #   ")"               -> cierre.
        #   "->"              -> retorno.
        #   "Image.Image"     -> tipo.
        #   ":"               -> bloque.
        # PARA QUÉ: decodificar bytes a PIL.Image normalizado.
        # POR QUÉ: desacopla la red del decodificado (útil en tests offline).
        # QUÉ HACE: BytesIO(data) -> Image.open -> normaliza a RGB.
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb para probar
        #   procesadores sin depender de internet (cargas bytes desde disco).
        # SI SE QUITA: los tests dependerían de requests (frágiles).
        # TRAMPA: "¿Y si data está vacío?" -> Image.open lanza UnidentifiedImageError.
        return ImageLoader._normalizar(Image.open(BytesIO(data)))
        # QUÉ SIGNIFICA: "ImageLoader" + "." + "_normalizar" + "(Image.open(BytesIO(data)))".
        # PARA QUÉ: reutilizar la normalización RGBA/RGB (DRY).
        # POR QUÉ: load() y load_from_bytes() deben devolver lo mismo.
        # QUÉ HACE: decodifica y delega la normalización.
        # SE UTILIZA PARA: Se utiliza para asegurar que ambas rutas de carga
        #   (desde fuente y desde bytes) devuelvan SIEMPRE RGB.
        # SI SE QUITA: duplicarías la lógica de transparencia.
        # TRAMPA: "¿Por qué no llamar a load()?" -> load() recibe str, no bytes.

    @staticmethod
    # QUÉ SIGNIFICA: "@" + "staticmethod".
    # PARA QUÉ: load_local() no recibe self.
    # POR QUÉ: utilidad pura.
    # QUÉ HACE: convierte load_local() en método estático.
    # SE UTILIZA PARA: Se utiliza en notebooks para cargar imágenes desde una
    #   ruta local conocida, con mejor mensaje de error si la ruta no existe.
    # SI SE QUITA: tendrías que llamar load() y capturar FileNotFoundError tú.
    def load_local(path: str) -> Image.Image:
        # QUÉ SIGNIFICA:
        #   "def"         -> define.
        #   "load_local"  -> nombre descriptivo.
        #   "("           -> apertura.
        #   "path"        -> parámetro.
        #   ":"           -> separador.
        #   "str"         -> tipo.
        #   ")"           -> cierre.
        #   "->"          -> retorno.
        #   "Image.Image" -> tipo.
        #   ":"           -> bloque.
        # PARA QUÉ: cargar explícitamente desde disco.
        # POR QUÉ: fail-fast con mensaje claro si no existe.
        # QUÉ HACE: valida os.path.isfile -> Image.open -> normaliza.
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb cuando se
        #   trabaja con imágenes descargadas manualmente o fixtures locales.
        # SI SE QUITA: FileNotFoundError menos claro.
        # TRAMPA: "¿load() ya soporta rutas?" -> Sí, pero load_local() es
        #   explícito y valida antes (mejor DX en tests).
        if not os.path.isfile(path):
            # QUÉ SIGNIFICA: "if" + "not" + "os.path.isfile(path)" + ":".
            # PARA QUÉ: validar existencia antes de abrir.
            # POR QUÉ: mensaje de error claro en lugar de FileNotFoundError crudo.
            # QUÉ HACE: True si la ruta no existe o no es archivo.
            # SE UTILIZA PARA: Se utiliza para fallar temprano en tests con
            #   un mensaje que indique la ruta exacta faltante.
            # SI SE QUITA: Pillow lanzaría FileNotFoundError sin contexto.
            raise FileNotFoundError(f"No existe el archivo: {path}")
            # QUÉ SIGNIFICA: "raise" + "FileNotFoundError" + "(f"...")".
            # PARA QUÉ: lanzar error semántico.
            # POR QUÉ: FileNotFoundError es el tipo correcto para esto.
            # QUÉ HACE: interrumpe load_local() con mensaje claro.
            # SE UTILIZA PARA: Se utiliza para que el notebook/test muestre
            #   exactamente qué ruta falló.
            # SI SE QUITA: el error sería menos informativo.
        return ImageLoader._normalizar(Image.open(path))
        # QUÉ SIGNIFICA: "ImageLoader._normalizar(Image.open(path))".
        # PARA QUÉ: reutilizar normalización.
        # QUÉ HACE: abre y normaliza a RGB.
        # SE UTILIZA PARA: Se utiliza para devolver RGB consistente.
        # SI SE QUITA: duplicarías la lógica de alfa.

    @staticmethod
    # QUÉ SIGNIFICA: "@" + "staticmethod".
    # PARA QUÉ: _normalizar() no recibe self.
    # POR QUÉ: helper puro de normalización.
    # QUÉ HACE: convierte la imagen a método estático privado.
    # SE UTILIZA PARA: Se utiliza internamente por load(), load_from_bytes()
    #   y load_local() para garantizar que TODAS devuelvan RGB.
    # SI SE QUITA: cada método repetiría el bloque RGBA/RGB.
    def _normalizar(img: Image.Image) -> Image.Image:
        # QUÉ SIGNIFICA:
        #   "def"          -> define.
        #   "_normalizar"  -> nombre privado (convención "_").
        #   "("            -> apertura.
        #   "img"          -> parámetro.
        #   ":"            -> separador.
        #   "Image.Image"  -> tipo.
        #   ")"            -> cierre.
        #   "->"           -> retorno.
        #   "Image.Image"  -> tipo.
        #   ":"            -> bloque.
        # PARA QUÉ: centralizar la conversión a RGB (DRY).
        # POR QUÉ: load(), load_from_bytes() y load_local() comparten esta lógica.
        # QUÉ HACE: si tiene alfa/paleta -> compone sobre blanco; si no -> convert("RGB").
        # SE UTILIZA PARA: Se utiliza en las tres rutas de carga para garantizar
        #   que el resultado sea siempre PIL.Image en modo RGB.
        # SI SE QUITA: se duplicaría el bloque RGBA/RGB en 3 lugares.
        # TRAMPA: "¿Por qué _normalizar y no normalizar?" -> "_" indica que es
        #   un helper interno, no parte de la API pública.
        if img.mode in ("RGBA", "LA", "P"):
            # QUÉ SIGNIFICA: pertenencia de img.mode a la tupla de modos con alfa.
            # PARA QUÉ: detectar si necesita composición.
            # POR QUÉ: convert("RGB") descarta alfa.
            # QUÉ HACE: True si necesita composición.
            # SE UTILIZA PARA: Se utiliza para decidir la rama de composición.
            # SI SE QUITA: PNGs transparentes con fondo negro.
            img = img.convert("RGBA")
            # Convierte a RGBA (garantiza último canal = alfa).
            fondo = Image.new("RGB", img.size, (255, 255, 255))
            # Crea lienzo blanco del mismo tamaño.
            fondo.paste(img, mask=img.split()[-1])
            # Compone usando el alfa como máscara.
            return fondo
            # Devuelve la imagen compuesta.
        return img.convert("RGB")
        # Si no había alfa, convierte directo a RGB.

    @staticmethod
    # QUÉ SIGNIFICA: "@" + "staticmethod".
    # PARA QUÉ: save() no recibe self.
    # POR QUÉ: utilidad pura de I/O.
    # QUÉ HACE: convierte save() en método estático.
    # SE UTILIZA PARA: Se utiliza en notebooks y en el menú para guardar
    #   resultados en disco de forma segura (crea directorios, sanitiza nombre).
    # SI SE QUITA: tendrías que gestionar os.makedirs manualmente.
    def save(img: Image.Image, path: str) -> str:
        # QUÉ SIGNIFICA:
        #   "def"         -> define.
        #   "save"        -> nombre.
        #   "("           -> apertura.
        #   "img"         -> parámetro.
        #   ":"           -> separador.
        #   "Image.Image" -> tipo.
        #   ","           -> separador.
        #   "path"        -> parámetro.
        #   ":"           -> separador.
        #   "str"         -> tipo.
        #   ")"           -> cierre.
        #   "->"          -> retorno.
        #   "str"         -> tipo (devuelve la ruta efectivamente guardada).
        #   ":"           -> bloque.
        # PARA QUÉ: guardar PIL.Image en disco creando directorios.
        # POR QUÉ: evita FileNotFoundError si el directorio no existe.
        # QUÉ HACE: sanitiza nombre, crea dir, guarda, devuelve ruta.
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb para guardar
        #   resultados de procesadores y compararlos manualmente.
        # SI SE QUITA: tendrías que repetir os.makedirs + img.save en cada test.
        # TRAMPA: "¿save devuelve algo?" -> Sí, la ruta, para encadenar logs.
        dir_destino = os.path.dirname(os.path.abspath(path))
        # QUÉ SIGNIFICA: "dir_destino" + "=" + "os.path.dirname(os.path.abspath(path))".
        # PARA QUÉ: obtener el directorio absoluto del destino.
        # POR QUÉ: os.makedirs necesita el directorio, no el archivo.
        # QUÉ HACE: extrae el directorio del path.
        # SE UTILIZA PARA: Se utiliza para crear el directorio si no existe.
        # SI SE QUITA: os.makedirs no tendría destino.
        if dir_destino and not os.path.isdir(dir_destino):
            # QUÉ SIGNIFICA: "if" + "dir_destino" + "and not os.path.isdir(...)" + ":".
            # PARA QUÉ: crear directorio solo si falta.
            # POR QUÉ: os.makedirs(exist_ok=True) evita error si existe.
            # QUÉ HACE: True si hay directorio y no existe.
            # SE UTILIZA PARA: Se utiliza para crear el directorio antes de guardar.
            # SI SE QUITA: img.save fallaría si el dir no existe.
            os.makedirs(dir_destino, exist_ok=True)
            # Crea el directorio (y padres) si no existe.
        img.save(path)
        # Guarda la imagen en disco (Pillow infiere formato por extensión).
        return path
        # Devuelve la ruta para encadenar logs ("guardado en: ...").


# ===========================================================================
# 5. ImageVisualizer
# ===========================================================================

class ImageVisualizer:
    # QUÉ SIGNIFICA: "class" + "ImageVisualizer" + ":".
    # PARA QUÉ: comparar y guardar imágenes.
    # POR QUÉ: SRP (visualizar != procesar).
    # QUÉ HACE: define la clase.
    # SE UTILIZA PARA: Se utiliza en las opciones 2, 3, 4 y 5 del menú para
    #   mostrar la comparación original/procesada y guardarla como PNG.
    # SI SE QUITA: no habría comparación ni archivo de salida.
    """Renderiza comparaciones y las exporta a disco. [SRP]"""

    @staticmethod
    # QUÉ SIGNIFICA: "@" + "staticmethod".
    # PARA QUÉ: ensure_agg_backend() no recibe self.
    # POR QUÉ: utilidad pura de configuración global de matplotlib.
    # QUÉ HACE: convierte el método en estático.
    # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb al inicio para
    #   evitar que plt.show() abra ventanas en entornos headless (Colab,
    #   servidores, GitHub Actions).
    # SI SE QUITA: plt.show() podría fallar o bloquear en headless.
    def ensure_agg_backend() -> None:
        # QUÉ SIGNIFICA: "def" + "ensure_agg_backend" + "()" + "->" + "None" + ":".
        # PARA QUÉ: forzar backend Agg (no interactivo) si no hay DISPLAY.
        # POR QUÉ: Agg permite savefig en headless sin abrir ventanas.
        # QUÉ HACE: si matplotlib no tiene backend interactivo, cambia a Agg.
        # SE UTILIZA PARA: Se utiliza al inicio del notebook y en entornos CI.
        # SI SE QUITA: plt.show() podría lanzar warnings o bloquear.
        # TRAMPA: "¿Por qué no siempre Agg?" -> En local quieres ver la ventana.
        import matplotlib
        # Import local para no forzar matplotlib.use() al importar el módulo.
        if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
            # QUÉ SIGNIFICA: "if" + "os.environ.get('DISPLAY','')==''" + "and" + "os.name != 'nt'".
            # PARA QUÉ: detectar entorno sin pantalla (Linux/macOS headless).
            # POR QUÉ: en Windows no hay DISPLAY pero sí GUI.
            # QUÉ HACE: True si estamos en headless Unix.
            # SE UTILIZA PARA: Se utiliza para decidir si forzar Agg.
            # SI SE QUITA: plt.show() podría fallar en servidores.
            matplotlib.use("Agg", force=True)
            # Fuerza backend no interactivo.
        return None
        # Devuelve None explícitamente (claridad).

    @staticmethod
    # QUÉ SIGNIFICA: "@" + "staticmethod".
    # PARA QUÉ: compare() no recibe self.
    # POR QUÉ: no usa estado de instancia.
    # QUÉ HACE: convierte compare() en método estático.
    # SE UTILIZA PARA: Se utiliza para poder llamar
    #   ImageVisualizer.compare(result) sin instanciar la clase.
    # SI SE QUITA: tendrías que crear ImageVisualizer() (antipatrón).
    def compare(result: ProcessResult) -> None:
        # QUÉ SIGNIFICA:
        #   "def"             -> define.
        #   "compare"         -> nombre.
        #   "("               -> apertura.
        #   "result"          -> parámetro.
        #   ":"               -> separador.
        #   "ProcessResult"   -> tipo.
        #   ")"               -> cierre.
        #   "->"              -> retorno.
        #   "None"            -> tipo None.
        #   ":"               -> bloque.
        # PARA QUÉ: comparar, guardar, mostrar y liberar.
        # POR QUÉ: recibe ProcessResult (menos acoplamiento que 5 parámetros).
        # QUÉ HACE: ejecuta los 6 pasos de visualización.
        # SE UTILIZA PARA: Se utiliza en FotoApp.menu opciones 2, 3, 4 y 5 para
        #   tomar el ProcessResult que devuelve cada procesador y generar la
        #   comparación visual + el archivo PNG.
        # SI SE QUITA: no habría comparación ni archivo.
        # TRAMPA: "¿Por qué ProcessResult y no 5 params?" -> Añadir campo no rompe firma.

        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        # QUÉ SIGNIFICA:
        #   "_"             -> descarta Figure.
        #   ","             -> separador.
        #   "ax"            -> ndarray shape (2,).
        #   "="             -> asignación.
        #   "plt"           -> módulo.
        #   "."             -> acceso.
        #   "subplots"      -> función.
        #   "("             -> apertura.
        #   "1"             -> filas.
        #   ","             -> separador.
        #   "2"             -> columnas.
        #   ","             -> separador.
        #   "figsize"       -> keyword.
        #   "="             -> asignación.
        #   "("             -> apertura tupla.
        #   "10"            -> ancho pulgadas.
        #   ","             -> separador.
        #   "5"             -> alto pulgadas.
        #   ")"             -> cierre tupla.
        #   ")"             -> cierre.
        # PARA QUÉ: crear figura 1x2 de 10x5 pulgadas (~1000x500 px a 100 DPI).
        # POR QUÉ: "_" descarta Figure (no se usa).
        # QUÉ HACE: crea la figura y los dos ejes.
        # SE UTILIZA PARA: Se utiliza para crear el lienzo donde se van a dibujar
        #   las dos imágenes (original y procesada) lado a lado.
        # SI SE QUITA: no habría layout comparativo.
        # TRAMPA: "¿Por qué descartar Figure?" -> Todo es sobre ax.

        ax[0].imshow(result.original); ax[0].set_title(result.titulo_izq); ax[0].axis("off")
        # QUÉ SIGNIFICA:
        #   "ax"                     -> ndarray.
        #   "[0]"                    -> primer eje.
        #   ".imshow"                -> método.
        #   "(result.original)"      -> imagen.
        #   ";"                      -> separador de sentencias.
        #   "ax[0]"                  -> primer eje.
        #   ".set_title"             -> método.
        #   "(result.titulo_izq)"    -> título.
        #   ";"                      -> separador.
        #   "ax[0]"                  -> primer eje.
        #   ".axis"                  -> método.
        #   "("off")"                -> argumento.
        # PARA QUÉ: dibujar original con título, sin ejes.
        # POR QUÉ: axis("off") oculta ticks (más limpio).
        # QUÉ HACE: dibuja la imagen izquierda.
        # SE UTILIZA PARA: Se utiliza para dibujar la imagen original en el panel
        #   izquierdo con el título "Original".
        # SI SE QUITA: se verían ejes y sin título.
        ax[1].imshow(result.procesada); ax[1].set_title(result.titulo_der); ax[1].axis("off")
        # QUÉ SIGNIFICA: análogo con "[1]" (panel derecho).
        # PARA QUÉ: dibujar procesada con título, sin ejes.
        # POR QUÉ: el usuario debe ver el resultado del procesamiento.
        # QUÉ HACE: dibuja la imagen derecha.
        # SE UTILIZA PARA: Se utiliza para dibujar la imagen procesada en el
        #   panel derecho con el título "Contraste", "Blur", "Canny", etc.
        # SI SE QUITA: no se vería el resultado del procesamiento.

        plt.tight_layout()
        # QUÉ SIGNIFICA: "plt" + "." + "tight_layout" + "()".
        # PARA QUÉ: ajustar márgenes.
        # POR QUÉ: evita solapamiento de títulos.
        # QUÉ HACE: recalcula márgenes automáticamente.
        # SE UTILIZA PARA: Se utiliza para que los títulos de los dos paneles no
        #   se solapen ni se corten cuando se guarden en el PNG.
        # SI SE QUITA: títulos cortados.

        plt.savefig(result.archivo, bbox_inches="tight")
        # QUÉ SIGNIFICA:
        #   "plt"            -> módulo.
        #   "."              -> acceso.
        #   "savefig"        -> método.
        #   "("              -> apertura.
        #   "result.archivo" -> nombre.
        #   ","              -> separador.
        #   "bbox_inches"    -> keyword.
        #   "="              -> asignación.
        #   ""tight""        -> valor.
        #   ")"              -> cierre.
        # PARA QUÉ: guardar PNG recortando blanco.
        # POR QUÉ: ANTES de show() (show puede consumir figura).
        # QUÉ HACE: escribe el archivo PNG en disco.
        # SE UTILIZA PARA: Se utiliza para guardar la comparación como archivo
        #   PNG en el disco, con el nombre indicado en result.archivo.
        # SI SE QUITA: no se guardaría archivo.
        # TRAMPA: "¿Qué DPI?" -> rcParams['savefig.dpi'] (100 por defecto).

        plt.show()
        # QUÉ SIGNIFICA: "plt" + "." + "show" + "()".
        # PARA QUÉ: mostrar ventana con la comparación.
        # POR QUÉ: feedback visual al usuario.
        # QUÉ HACE: abre ventana interactiva (bloquea solo en backends interactivos).
        # SE UTILIZA PARA: Se utiliza para que el usuario vea la comparación en
        #   una ventana mientras el programa está en ejecución.
        # SI SE QUITA: el usuario no vería la comparación.
        # TRAMPA: "¿Bloquea siempre?" -> No; en Agg (headless) es no-op.

        plt.close()
        # QUÉ SIGNIFICA: "plt" + "." + "close" + "()".
        # PARA QUÉ: liberar la figura de la memoria.
        # POR QUÉ: sin close(), en bucle, plt.get_fignums() crece -> RAM.
        # QUÉ HACE: elimina la figura de matplotlib.
        # SE UTILIZA PARA: Se utiliza para liberar la figura después de mostrarla
        #   y guardarla.
        # SI SE QUITA: acumulación de figuras.
        # TRAMPA: "¿Memory leak?" -> No clásico; acumulación evitable.

    @staticmethod
    # QUÉ SIGNIFICA: "@" + "staticmethod".
    # PARA QUÉ: show_single() no recibe self.
    # POR QUÉ: utilidad pura de visualización.
    # QUÉ HACE: convierte show_single() en método estático.
    # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb para ver UNA sola
    #   imagen (por ejemplo, la original recién cargada o un resultado suelto).
    # SI SE QUITA: solo podrías ver pares original/procesada.
    def show_single(img: Image.Image, titulo: str = "Imagen", archivo: Optional[str] = None) -> None:
        # QUÉ SIGNIFICA:
        #   "def"           -> define.
        #   "show_single"   -> nombre.
        #   "("             -> apertura.
        #   "img"           -> parámetro.
        #   ":"             -> separador.
        #   "Image.Image"   -> tipo.
        #   ","             -> separador.
        #   "titulo"        -> parámetro con default.
        #   ":"             -> separador.
        #   "str"           -> tipo.
        #   "="             -> default.
        #   ""Imagen""      -> valor por defecto.
        #   ","             -> separador.
        #   "archivo"       -> parámetro opcional.
        #   ":"             -> separador.
        #   "Optional[str]" -> tipo (puede ser None).
        #   ")"             -> cierre.
        #   "->"            -> retorno.
        #   "None"          -> tipo None.
        #   ":"             -> bloque.
        # PARA QUÉ: mostrar una sola imagen con título y guardado opcional.
        # POR QUÉ: útil para inspeccionar resultados sueltos en el notebook.
        # QUÉ HACE: crea figura 1x1, imshow, título, savefig opcional, show, close.
        # SE UTILIZA PARA: Se utiliza en tests para verificar visualmente que
        #   una imagen se cargó o procesó correctamente.
        # SI SE QUITA: no podrías ver imágenes sueltas.
        # TRAMPA: "¿Por qué no usar compare?" -> compare requiere 2 imágenes.
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        # Crea figura 1x1 de 6x6 pulgadas.
        ax.imshow(img)
        # Dibuja la imagen.
        ax.set_title(titulo)
        # Aplica el título.
        ax.axis("off")
        # Oculta los ejes.
        plt.tight_layout()
        # Ajusta márgenes.
        if archivo:
            # Solo guarda si se pidió archivo.
            plt.savefig(archivo, bbox_inches="tight")
            # Guarda la imagen en disco.
        plt.show()
        # Muestra la ventana (no-op en Agg).
        plt.close()
        # Libera la figura.


# ===========================================================================
# 6. ContrastProcessor
# ===========================================================================

class ContrastProcessor(ImageProcessor):
    # QUÉ SIGNIFICA: "class" + "ContrastProcessor" + "(ImageProcessor)" + ":".
    # PARA QUÉ: implementar contraste por ecualización de Y.
    # POR QUÉ: hereda el contrato de ImageProcessor.
    # QUÉ HACE: define el procesador.
    # SE UTILIZA PARA: Se utiliza en la opción 2 del menú para mejorar el
    #   contraste de la imagen ecualizando solo el canal de luminancia.
    # SI SE QUITA: no habría opción de contraste.
    """Contraste por ecualización del canal Y (YCrCb)."""

    def process(self, img: Image.Image) -> ProcessResult:
        # QUÉ SIGNIFICA: "def" + "process" + "(self, img)" + "->" + "ProcessResult" + ":".
        # PARA QUÉ: implementar el contrato abstracto.
        # POR QUÉ: polimorfismo: FotoApp lo usa sin saber la clase concreta.
        # QUÉ HACE: ecualiza Y y devuelve ProcessResult.
        # SE UTILIZA PARA: Se utiliza desde FotoApp.menu opción 2:
        #   self._contrast.process(self.img).
        # SI SE QUITA: no habría mejora de contraste.

        i = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # PIL RGB -> ndarray BGR (formato que OpenCV espera).

        y = cv2.cvtColor(i, cv2.COLOR_BGR2YCrCb)
        # BGR -> YCrCb: y[:,:,0]=Y, y[:,:,1]=Cr, y[:,:,2]=Cb.

        y[:, :, 0] = cv2.equalizeHist(y[:, :, 0])
        # Ecualiza SOLO el canal de luminancia (Y) in-place.

        res = Image.fromarray(cv2.cvtColor(cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR), cv2.COLOR_BGR2RGB))
        # YCrCb -> BGR -> RGB -> PIL.

        return ProcessResult(img, res, "Original", "Contraste", "c.png")
        # Empaqueta el contrato de 5 campos.


# ===========================================================================
# 7. FilterProcessor
# ===========================================================================

class FilterProcessor(ImageProcessor):
    # QUÉ SIGNIFICA: "class" + "FilterProcessor" + "(ImageProcessor)" + ":".
    # PARA QUÉ: aplicar filtros espaciales de Pillow.
    # POR QUÉ: hereda el contrato.
    # QUÉ HACE: define el procesador.
    # SE UTILIZA PARA: Se utiliza en la opción 3 del menú para aplicar uno de
    #   los nueve filtros de Pillow a la imagen actual.
    # SI SE QUITA: no habría opción de filtros.
    """Filtros espaciales de Pillow."""

    FILTROS_VALIDOS = {
        "blur": ImageFilter.BLUR,
        "contour": ImageFilter.CONTOUR,
        "detail": ImageFilter.DETAIL,
        "edge enhance": ImageFilter.EDGE_ENHANCE,
        "edge enhance more": ImageFilter.EDGE_ENHANCE_MORE,
        "emboss": ImageFilter.EMBOSS,
        "find edges": ImageFilter.FIND_EDGES,
        "sharpen": ImageFilter.SHARPEN,
        "smooth": ImageFilter.SMOOTH,
    }

    def __init__(self, filter_name: str):
        # QUÉ SIGNIFICA: constructor con validación fail-fast.
        # PARA QUÉ: validar y guardar normalizado.
        # POR QUÉ: fallar al crear es mejor que fallar tras hacer trabajo.
        # QUÉ HACE: valida el nombre y lo guarda.
        # SE UTILIZA PARA: Se utiliza desde FotoApp.menu opción 3 a través del
        #   factory inyectado: processor = self._filter_factory(nom).
        # SI SE QUITA: process() haría dict[clave_inválida] -> KeyError confuso.
        # TRAMPA: "¿Por qué validar en __init__?" -> Fail-fast.

        if filter_name.lower() not in self.FILTROS_VALIDOS:
            # Comprueba si el nombre no es válido.
            raise ValueError(f"Filtro no válido. Opciones: {list(self.FILTROS_VALIDOS.keys())}")
            # Lanza error capturado por except Exception del menú.
        self.filter_name = filter_name.lower()
        # Guarda normalizado para usarlo en process().

    def process(self, img: Image.Image) -> ProcessResult:
        # QUÉ SIGNIFICA: implementar el contrato.
        # PARA QUÉ: aplicar el filtro y devolver ProcessResult.
        # POR QUÉ: polimorfismo.
        # QUÉ HACE: aplica el filtro y devuelve ProcessResult.
        # SE UTILIZA PARA: Se utiliza desde FotoApp.menu opción 3.
        # SI SE QUITA: no habría filtro aplicado.

        filtered = img.filter(self.FILTROS_VALIDOS[self.filter_name])
        # Aplica la convolución del filtro.

        archivo = f"f_{_safe_filename(self.filter_name)}.png"
        # Genera "f_blur.png", "f_edge_enhance.png", etc.

        return ProcessResult(img, filtered, "Original", self.filter_name.capitalize(), archivo)
        # Empaqueta el contrato de 5 campos.


# ===========================================================================
# 8. EdgeDetectionProcessor
# ===========================================================================

class EdgeDetectionProcessor(ImageProcessor):
    # QUÉ SIGNIFICA: "class" + "EdgeDetectionProcessor" + "(ImageProcessor)" + ":".
    # PARA QUÉ: comparar Canny vs Laplaciano.
    # POR QUÉ: didáctico: dos enfoques de detección de bordes.
    # QUÉ HACE: define el procesador.
    # SE UTILIZA PARA: Se utiliza en la opción 4 del menú para comparar dos
    #   algoritmos de detección de bordes sobre la misma imagen.
    # SI SE QUITA: no habría opción de bordes clásicos.
    """Compara Canny vs Laplaciano."""

    def process(self, img: Image.Image) -> ProcessResult:
        # QUÉ SIGNIFICA: implementar el contrato.
        # PARA QUÉ: detectar bordes con 2 algoritmos.
        # POR QUÉ: polimorfismo.
        # QUÉ HACE: detecta bordes con Canny y Laplaciano.
        # SE UTILIZA PARA: Se utiliza desde FotoApp.menu opción 4.
        # SI SE QUITA: no habría comparación de bordes.

        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        # Convierte a gris (1 canal) para calcular gradientes.

        canny = cv2.Canny(g, 100, 200)
        # Canny: gaussiano + Sobel + supresión + histéresis.

        laplacian = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_16S))
        # Laplaciano: |nabla^2| -> uint8 (0-255).

        return ProcessResult(canny, laplacian, "Canny", "Laplaciano", "b.png")
        # Empaqueta el contrato de 5 campos.


# ===========================================================================
# 9. EJERCICIO 4 — BOCETO DE LÍNEA (>=2 técnicas)
# ===========================================================================

def boceto_linea(img: Image.Image, tecnica: str = "canny",
                 umbral1: int = 100, umbral2: int = 200) -> Image.Image:
    # [EJERCICIO 4] FUNCIÓN NUEVA
    # QUÉ SIGNIFICA:
    #   "def" + "boceto_linea" + "(img, tecnica, umbral1, umbral2)" + "->" + "Image.Image".
    # PARA QUÉ: devolver un boceto de línea (bordes realzados) de la imagen.
    # POR QUÉ: el enunciado pide una función que reciba la imagen y devuelva
    #   otra imagen tipo boceto (líneas negras sobre blanco o al revés).
    # QUÉ HACE: convierte a gris, aplica la técnica elegida, binariza e invierte.
    # SE UTILIZA PARA: Se utiliza en el notebook y en comparar_bocetos() para
    #   generar los bocetos de las 3 técnicas.
    # SI SE QUITA: no se cumpliría el punto 4 del enunciado.
    # TRAMPA: "¿umbral1/umbral2?" -> parámetros ajustables de Canny; 100/200 es
    #   la heurística clásica 1:2 (no universal, pero razonable).
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    # Convierte a escala de grises (los bordes se calculan sobre intensidad).
    if tecnica == "canny":
        # Canny: 1ra derivada + histéresis (robusto, bordes finos y conectados).
        b = cv2.Canny(g, umbral1, umbral2)
    elif tecnica == "sobel":
        # Sobel: gradiente en X e Y; magnitud combinada.
        sx = cv2.Sobel(g, cv2.CV_16S, 1, 0)
        sy = cv2.Sobel(g, cv2.CV_16S, 0, 1)
        mag = cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5,
                              cv2.convertScaleAbs(sy), 0.5, 0)
        _, b = cv2.threshold(mag, umbral1, 255, cv2.THRESH_BINARY)
    elif tecnica == "laplaciano":
        # Laplaciano: 2da derivada (isotrópico, más sensible al ruido).
        lap = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_16S))
        _, b = cv2.threshold(lap, umbral1, 255, cv2.THRESH_BINARY)
    elif tecnica == "umbral":
        # Umbralización de Otsu sobre la imagen gris (boceto "silueta").
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Técnica desconocida: fallar claro.
        raise ValueError(
            f"Técnica no válida: {tecnica!r}. "
            f"Opciones: canny, sobel, laplaciano, umbral"
        )
    b = cv2.bitwise_not(b)
    # Invierte: líneas negras sobre fondo blanco (estilo boceto).
    return Image.fromarray(b).convert("RGB")
    # Devuelve PIL.Image en RGB (consistente con el resto del pipeline).


def comparar_bocetos(img: Image.Image, archivo: str = "bocetos.png") -> None:
    # [EJERCICIO 4] FUNCIÓN NUEVA
    # QUÉ SIGNIFICA: "def" + "comparar_bocetos" + "(img, archivo)" + "->" + "None".
    # PARA QUÉ: mostrar original + 3 técnicas de boceto lado a lado y guardar.
    # POR QUÉ: el enunciado pide "mostrar los resultados lado a lado".
    # QUÉ HACE: crea figura 1x4 (original + canny + sobel + umbral), guarda y cierra.
    # SE UTILIZA PARA: Se utiliza en el notebook (punto 4) para la evidencia visual.
    # SI SE QUITA: no habría comparación lado a lado del boceto.
    tecnicas = ["canny", "sobel", "umbral"]
    # 3 técnicas a comparar (cumple "al menos dos" del enunciado).
    fig, axes = plt.subplots(1, len(tecnicas) + 1,
                             figsize=(4 * (len(tecnicas) + 1), 4))
    # Figura 1x4 de 16x4 pulgadas.
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    # Panel izquierdo: imagen original.
    for ax, t in zip(axes[1:], tecnicas):
        # Itera sobre los 3 paneles restantes y las 3 técnicas.
        ax.imshow(boceto_linea(img, t))
        # Dibuja el boceto de la técnica t.
        ax.set_title(t.capitalize())
        # Título: Canny, Sobel, Umbral.
        ax.axis("off")
        # Sin ejes.
    plt.tight_layout()
    # Ajusta márgenes.
    plt.savefig(archivo, bbox_inches="tight")
    # Guarda el PNG de comparación.
    plt.show()
    # Muestra (no-op en Agg).
    plt.close()
    # Libera la figura.


class BocetoProcessor(ImageProcessor):
    # [EJERCICIO 4] CLASE NUEVA
    # QUÉ SIGNIFICA: "class" + "BocetoProcessor" + "(ImageProcessor)" + ":".
    # PARA QUÉ: integrar el boceto de línea al patrón procesador (polimorfismo).
    # POR QUÉ: FotoApp usa ImageProcessor; así la opción 5 del menú funciona
    #   igual que las demás.
    # QUÉ HACE: devuelve ProcessResult con Original vs Boceto (Canny).
    # SE UTILIZA PARA: Se utiliza desde FotoApp.menu opción 5 (Boceto).
    # SI SE QUITA: la opción 5 no tendría procesador polimórfico.
    """Boceto de línea (Canny) como procesador."""

    def process(self, img: Image.Image) -> ProcessResult:
        # QUÉ SIGNIFICA: implementar el contrato.
        # PARA QUÉ: devolver Original vs Boceto.
        # QUÉ HACE: llama a boceto_linea(img, "canny").
        # SE UTILIZA PARA: Se utiliza desde FotoApp.detectar_boceto() y menu opción 5.
        # SI SE QUITA: no habría procesador de boceto.
        res = boceto_linea(img, "canny")
        # Genera el boceto con Canny (líneas negras sobre blanco).
        return ProcessResult(img, res, "Original", "Boceto (Canny)", "boceto_canny.png")
        # Empaqueta el contrato de 5 campos.


# ===========================================================================
# 10. FotoApp (controlador con DIP real)
# ===========================================================================

class FotoApp:
    # QUÉ SIGNIFICA: "class" + "FotoApp" + ":".
    # PARA QUÉ: controlador: estado, menú, orquestación.
    # POR QUÉ: DIP real vía inyección de dependencias.
    # QUÉ HACE: define el controlador.
    # SE UTILIZA PARA: Se utiliza en __main__ para crear la instancia que
    #   arranca el menú interactivo, y en el notebook para llamar a los métodos
    #   programáticos (cargar_imagen, aplicar_contraste, etc.).
    # SI SE QUITA: no habría aplicación.
    """Controlador: estado, menú y orquestación. Aplica DIP vía inyección."""

    def __init__(
        self,
        contrast_processor: ImageProcessor,
        filter_factory: Callable[[str], ImageProcessor],
        edge_processor: ImageProcessor,
        boceto_processor: Optional[ImageProcessor] = None,
    ):
        # [EJERCICIO 4] Se añade "boceto_processor" (opcional para no romper
        # compatibilidad con instanciaciones previas sin ese argumento).
        # QUÉ SIGNIFICA: constructor con 4 dependencias tipadas (la 4ª opcional).
        # PARA QUÉ: inyectar dependencias (DIP real).
        # POR QUÉ: FotoApp NO instancia clases concretas; las recibe.
        # QUÉ HACE: guarda las dependencias y el estado inicial.
        # SE UTILIZA PARA: Se utiliza desde __main__ (composition root).
        # SI SE QUITA: violación DIP (instanciaría concretos).
        # TRAMPA: "¿Por qué Callable y no Type?" -> FilterProcessor requiere
        #   filter_name en __init__; Callable[[str], ImageProcessor] lo modela.
        self._contrast = contrast_processor
        # Guarda el procesador de contraste.
        self._filter_factory = filter_factory
        # Guarda el factory de filtros.
        self._edge = edge_processor
        # Guarda el procesador de bordes.
        self._boceto = boceto_processor if boceto_processor is not None else BocetoProcessor()
        # [EJERCICIO 4] Guarda el procesador de boceto (o uno por defecto).
        self.img: Optional[Image.Image] = None
        # Estado inicial: sin imagen.

    def _validar_imagen(self) -> None:
        # QUÉ SIGNIFICA: validar que self.img no sea None.
        # PARA QUÉ: DRY: centraliza la validación (opciones 2, 3, 4, 5).
        # POR QUÉ: el enunciado pide usar excepciones para validar.
        # QUÉ HACE: lanza ValueError si no hay imagen.
        # SE UTILIZA PARA: Se utiliza al inicio de las opciones 2, 3, 4 y 5.
        # SI SE QUITA: procesarías con img=None (AttributeError).
        if self.img is None:
            # Comprueba si no se cargó imagen.
            raise ValueError("Debe cargar y redimensionar una imagen primero (opción 1).")
            # Lanza error capturado por el menú.

    @staticmethod
    def _ajustar_a_red(img: Image.Image, red: str) -> Image.Image:
        # QUÉ SIGNIFICA: componer la imagen en lienzo EXACTO (letterbox).
        # PARA QUÉ: garantizar dimensiones exactas sin distorsión.
        # POR QUÉ: thumbnail solo respeta proporción; para exactas, lienzo fijo.
        # QUÉ HACE: crea un lienzo (w,h) exacto y pega la imagen centrada.
        # SE UTILIZA PARA: Se utiliza en la opción 1 del menú, en redimensionar()
        #   y en la función pública redimensionar_para_red().
        # SI SE QUITA: dimensiones no exactas (solo máximos).
        # TRAMPA: "¿Letterbox vs stretch?" -> Letterbox respeta proporción;
        #   stretch deforma; crop recorta.
        if red.lower() not in DIMENSIONES_REDES:
            # [EJERCICIO 1] Validación con excepción (fail-fast).
            raise ValueError(
                f"Red no válida: {red!r}. Opciones: {list(DIMENSIONES_REDES.keys())}"
            )
            # Lanza error claro con opciones válidas.
        w, h = DIMENSIONES_REDES[red.lower()]
        # Extrae dimensiones EXACTAS (w, h).
        copia = img.copy()
        # Copia para no mutar la original (thumbnail es in-place).
        copia.thumbnail((w, h), Image.Resampling.LANCZOS)
        # Reduce respetando proporción (no agranda).
        lienzo = Image.new("RGB", (w, h), (255, 255, 255))
        # Crea lienzo exacto de la red, fondo blanco.
        x = (w - copia.width) // 2
        # Centrado horizontal.
        y = (h - copia.height) // 2
        # Centrado vertical.
        lienzo.paste(copia, (x, y))
        # Pega la copia centrada (letterbox/pillarbox).
        return lienzo
        # Devuelve el lienzo exacto.

    # -----------------------------------------------------------------------
    # API PROGRAMÁTICA (usar desde notebook/tests sin input)
    # -----------------------------------------------------------------------

    def cargar_imagen(self, source: str, red: Optional[str] = None) -> Image.Image:
        # QUÉ SIGNIFICA: cargar + (opcional) redimensionar sin usar input().
        # PARA QUÉ: el notebook necesita llamar la lógica de la opción 1 sin menú.
        # QUÉ HACE: ImageLoader.load(source) -> _ajustar_a_red -> guarda self.img.
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb para preparar
        #   self.img antes de probar los procesadores.
        # SI SE QUITA: el notebook tendría que replicar la lógica del menú.
        img = ImageLoader.load(source)
        # Carga la imagen desde clave/URL/ruta.
        if red:
            # Solo redimensiona si se pidió red.
            img = self._ajustar_a_red(img, red)
            # Ajusta a dimensiones exactas (letterbox).
        self.img = img
        # Guarda en el estado.
        return self.img
        # Devuelve la imagen (para encadenar o imprimir).

    def redimensionar(self, red: str) -> Image.Image:
        # QUÉ SIGNIFICA: redimensionar la imagen actual a una red.
        # PARA QUÉ: cambiar de red sin recargar.
        # QUÉ HACE: valida self.img, aplica _ajustar_a_red, guarda.
        # SE UTILIZA PARA: Se utiliza en el notebook.
        # SI SE QUITA: tendrías que recargar la imagen para cambiar de red.
        self._validar_imagen()
        # Falla claro si no hay imagen.
        self.img = self._ajustar_a_red(self.img, red)
        # Ajusta a la nueva red.
        return self.img
        # Devuelve la imagen redimensionada.

    def aplicar_contraste(self) -> ProcessResult:
        # QUÉ SIGNIFICA: ejecutar la opción 2 sin menú.
        # PARA QUÉ: el notebook necesita invocar el procesador directamente.
        # QUÉ HACE: valida -> self._contrast.process(self.img).
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb.
        # SI SE QUITA: el notebook llamaría self._contrast.process(self.img).
        self._validar_imagen()
        # Falla claro si no hay imagen.
        return self._contrast.process(self.img)
        # Devuelve el ProcessResult.

    def aplicar_filtro(self, nombre: str) -> ProcessResult:
        # QUÉ SIGNIFICA: ejecutar la opción 3 sin menú.
        # PARA QUÉ: el notebook necesita invocar el factory + process.
        # QUÉ HACE: valida -> factory(nombre) -> process(self.img).
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb.
        # SI SE QUITA: el notebook replicaría la lógica del factory.
        self._validar_imagen()
        # Falla claro si no hay imagen.
        processor = self._filter_factory(nombre)
        # Crea el procesador vía factory inyectado (DIP).
        return processor.process(self.img)
        # Devuelve el ProcessResult.

    def detectar_bordes(self) -> ProcessResult:
        # QUÉ SIGNIFICA: ejecutar la opción 4 sin menú.
        # PARA QUÉ: el notebook necesita invocar el procesador de bordes.
        # QUÉ HACE: valida -> self._edge.process(self.img).
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb.
        # SI SE QUITA: el notebook llamaría self._edge.process(self.img).
        self._validar_imagen()
        # Falla claro si no hay imagen.
        return self._edge.process(self.img)
        # Devuelve el ProcessResult (Canny vs Laplaciano).

    def detectar_boceto(self) -> ProcessResult:
        # [EJERCICIO 4] MÉTODO NUEVO
        # QUÉ SIGNIFICA: ejecutar la opción 5 (boceto) sin menú.
        # PARA QUÉ: el notebook necesita invocar el procesador de boceto.
        # QUÉ HACE: valida -> self._boceto.process(self.img).
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb.
        # SI SE QUITA: el notebook llamaría self._boceto.process(self.img).
        self._validar_imagen()
        # Falla claro si no hay imagen.
        return self._boceto.process(self.img)
        # Devuelve el ProcessResult (Original vs Boceto Canny).

    def info(self) -> Dict[str, Any]:
        # QUÉ SIGNIFICA: exponer el estado actual (para debug y tests).
        # PARA QUÉ: en el notebook querrás verificar tamaño/modo sin tocar _img.
        # QUÉ HACE: devuelve dict con cargada, size, mode, dimensiones red.
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb para asserts.
        # SI SE QUITA: tendrías que acceder a app.img.size manualmente.
        if self.img is None:
            # Caso sin imagen cargada.
            return {"cargada": False, "size": None, "mode": None}
            # Devuelve estado vacío explícito.
        return {
            "cargada": True,
            "size": self.img.size,
            "mode": self.img.mode,
            "redes_disponibles": list(DIMENSIONES_REDES.keys()),
            "filtros_disponibles": list(FilterProcessor.FILTROS_VALIDOS.keys()),
        }
        # Devuelve estado completo (útil para imprimir en notebook).

    def guardar(self, path: str) -> str:
        # QUÉ SIGNIFICA: guardar self.img en disco.
        # PARA QUÉ: el notebook necesita exportar resultados.
        # QUÉ HACE: valida -> ImageLoader.save(self.img, path).
        # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb.
        # SI SE QUITA: llamarías ImageLoader.save(app.img, path) directamente.
        self._validar_imagen()
        # Falla claro si no hay imagen.
        return ImageLoader.save(self.img, path)
        # Guarda y devuelve la ruta.

    # -----------------------------------------------------------------------
    # MENÚ INTERACTIVO (usa la API programática internamente: DRY)
    # -----------------------------------------------------------------------

    def menu(self) -> None:
        # QUÉ SIGNIFICA: bucle interactivo.
        # PARA QUÉ: bucle interactivo.
        # POR QUÉ: -> None (no devuelve nada).
        # QUÉ HACE: muestra opciones y ejecuta.
        # SE UTILIZA PARA: Se utiliza en __main__: app.menu().
        # SI SE QUITA: no habría interfaz de usuario.
        while (op := input(
            "\n=== MENU FOTOAPP ===\n"
            "1: Cargar y Redimensionar\n"
            "2: Ajustar Contraste\n"
            "3: Aplicar Filtro (por número 1-9)\n"
            "4: Detección de Bordes (Canny vs Laplaciano)\n"
            "5: Boceto de Línea (Canny/Sobel/Umbral)\n"
            "6: Salir\n"
            "Seleccione: "
        )) != "6":
            # Bucle principal hasta opción 6 (Salir).
            try:
                # Bloque protegido: cualquier error cierra la iteración, no el programa.
                if op == "1":
                    # [EJERCICIO 1] Opción 1: cargar y redimensionar.
                    k = input("Clave o URL de la imagen: ")
                    # Lee la fuente (clave, URL o ruta).
                    red = input("Red destino (instagram/facebook/twitter/youtube): ")
                    # Lee la red destino.
                    self.img = redimensionar_para_red(k, red)
                    # Llama a la función pública del EJERCICIO 1.
                    print(f"¡Imagen cargada! Tamaño final: {self.img.size}")
                    # Feedback con tamaño final.
                elif op == "2":
                    # Opción 2: contraste por ecualización de Y.
                    ImageVisualizer.compare(self.aplicar_contraste())
                    # Procesa + compara.
                elif op == "3":
                    # [EJERCICIO 3] Opción 3: filtros por NÚMERO (1-9).
                    self._validar_imagen()
                    # Valida antes de mostrar filtros.
                    filtros_num = {
                        str(i + 1): n
                        for i, n in enumerate(FilterProcessor.FILTROS_VALIDOS)
                    }
                    # Mapeo número -> nombre (1: blur, 2: contour, ...).
                    for num, nombre in filtros_num.items():
                        print(f"  {num}: {nombre}")
                    # Muestra el menú numerado de filtros.
                    num = input("Número del filtro (1-9): ")
                    # Lee el número.
                    nom = filtros_num[num]
                    # Traduce número -> nombre (KeyError si inválido).
                    ImageVisualizer.compare(self.aplicar_filtro(nom))
                    # Procesa + compara.
                elif op == "4":
                    # Opción 4: bordes Canny vs Laplaciano.
                    ImageVisualizer.compare(self.detectar_bordes())
                    # Procesa + compara.
                elif op == "5":
                    # [EJERCICIO 4] Opción 5: boceto de línea.
                    self._validar_imagen()
                    # Valida antes de procesar.
                    print("Técnicas: canny, sobel, laplaciano, umbral")
                    # Muestra las técnicas disponibles.
                    tecnica = input("Técnica (Enter = canny): ").strip() or "canny"
                    # Lee la técnica (por defecto canny).
                    if tecnica not in ("canny", "sobel", "laplaciano", "umbral"):
                        # Valida la técnica elegida.
                        raise ValueError(f"Técnica no válida: {tecnica!r}")
                    res = boceto_linea(self.img, tecnica)
                    # Genera el boceto de la técnica elegida.
                    ImageVisualizer.compare(
                        ProcessResult(self.img, res, "Original",
                                      f"Boceto ({tecnica})", f"boceto_{tecnica}.png")
                    )
                    # Muestra y guarda la comparación.
                else:
                    # Opción inválida.
                    print("Opción no válida. Elija 1–6.")
                    # Feedback.
            except Exception as e:
                # Captura errores recuperables (ValueError, HTTPError, OSError).
                print("Error operativo:", e)
                # Imprime sin cerrar el programa.


# ===========================================================================
# 11. EJERCICIO 1 — FUNCIÓN PÚBLICA (ruta + palabra clave -> imagen ajustada)
# ===========================================================================

def redimensionar_para_red(ruta: str, red: str) -> Image.Image:
    # [EJERCICIO 1] FUNCIÓN NUEVA
    # QUÉ SIGNIFICA:
    #   "def" + "redimensionar_para_red" + "(ruta, red)" + "->" + "Image.Image".
    # PARA QUÉ: cumplir el EJERCICIO 1 del enunciado: recibe DOS strings (ruta
    #   y palabra clave de red) y devuelve la imagen redimensionada al tamaño
    #   EXACTO de la red, sin distorsionar (letterbox).
    # POR QUÉ: el enunciado pide UNA función con esa firma exacta.
    # QUÉ HACE:
    #   1) Valida que "red" sea una clave conocida (si no, ValueError).
    #   2) Carga la imagen con ImageLoader.load(ruta).
    #   3) La ajusta con FotoApp._ajustar_a_red(img, red).
    #   4) Devuelve la imagen resultante (PIL.Image en RGB).
    # SE UTILIZA PARA: Se utiliza en el menú (opción 1), en el notebook
    #   fotoapp_testdev.ipynb y en los tests.
    # SI SE QUITA: el requisito 1 del enunciado quedaría parcialmente cumplido.
    # TRAMPA: "¿Y si ruta es None?" -> ImageLoader.load(None) lanzará
    #   AttributeError en .lower(); mejor fallar que devolver basura.
    if red.lower() not in DIMENSIONES_REDES:
        # Valida la palabra clave ANTES de descargar la imagen (fail-fast).
        raise ValueError(
            f"Red no válida: {red!r}. Opciones: {list(DIMENSIONES_REDES.keys())}"
        )
        # Lanza error con las opciones válidas.
    img = ImageLoader.load(ruta)
    # Carga la imagen desde clave/URL/ruta local.
    return FotoApp._ajustar_a_red(img, red)
    # Aplica el letterbox exacto de la red y devuelve.


# ===========================================================================
# 12. EXPORTS PÚBLICOS
# ===========================================================================
__all__ = [
    # QUÉ SIGNIFICA: lista de nombres exportados por "from fotoapp import *".
    # PARA QUÉ: controlar qué se expone públicamente.
    # POR QUÉ: evita exportar imports internos (_safe_filename, os, re, etc.).
    # QUÉ HACE: restringe el wildcard import.
    # SE UTILIZA PARA: Se utiliza en fotoapp_testdev.ipynb.
    # SI SE QUITA: "import *" exportaría todo (incluidos os, re, cv2).
    # TRAMPA: "¿__all__ afecta imports explícitos?" -> No, solo al "*".
    "ENLACES",
    "DIMENSIONES_REDES",
    "ProcessResult",
    "ImageProcessor",
    "ImageLoader",
    "ImageVisualizer",
    "ContrastProcessor",
    "FilterProcessor",
    "EdgeDetectionProcessor",
    "BocetoProcessor",
    "FotoApp",
    "redimensionar_para_red",
    "boceto_linea",
    "comparar_bocetos",
]


# ===========================================================================
# 13. PUNTO DE ENTRADA (composition root)
# ===========================================================================
if __name__ == "__main__":
    # QUÉ SIGNIFICA: "if" + "__name__ == "__main__"" + ":".
    # PARA QUÉ: True si se ejecuta directamente (no al importar).
    # POR QUÉ: guarda de ejecución: permite reutilizar las clases sin lanzar el menú.
    # QUÉ HACE: decide si arrancar el menú.
    # SE UTILIZA PARA: Se utiliza para que el archivo se pueda importar.
    # SI SE QUITA: el menú se ejecutaría al importar (malo para tests).
    # TRAMPA: "¿Qué vale __name__?" -> "__main__" si directo; nombre del módulo si importado.
    ImageVisualizer.ensure_agg_backend()
    # Configura matplotlib para headless si hace falta.
    app = FotoApp(
        contrast_processor=ContrastProcessor(),
        filter_factory=FilterProcessor,
        edge_processor=EdgeDetectionProcessor(),
        boceto_processor=BocetoProcessor(),
    )
    # Composition root: ÚNICO lugar con clases concretas.
    app.menu()
    # Arranca el bucle interactivo.