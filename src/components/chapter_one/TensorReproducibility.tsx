import { Box, Text } from '@chakra-ui/react'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import chapterOne from '../../data/chapterOne'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'

const TensorReproducibility = () => {
  const [ 
    repro, rand, randpy, randf, pseudorand, pseudorandf, randseed, randseedpy
  ] = chapterOne[6].sections
  const [randomCode, reproCode] = pythonCode.slice(15, 17).map(obj => obj.code)
  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='lg' color='gray.600'>{repro.name}</Text>
        <Definition definition={repro}/>
      </Box>
      <Box py={5}>
        <Text as='b' fontSize='lg' color='gray.600'>{rand.name}</Text>
        <Definition definition={rand}/>
      </Box>
      <BaseGrid section={randpy}/>
      <CodeDisplay codes={randomCode}/>
      <BaseGrid section={randf}/>
      <Box py={5}>
        <Text as='b' fontSize='lg' color='gray.600'>{pseudorand.name}</Text>
        <Definition definition={pseudorand}/>
      </Box>
      <BaseGrid section={pseudorandf}/>
      <Box py={5}>
        <Text as='b' fontSize='lg' color='gray.600'>{randseed.name}</Text>
        <Definition definition={randseed}/>
      </Box>
      <BaseGrid section={randseedpy}/>
      <CodeDisplay codes={reproCode}/>
    </>
  )
}

export default TensorReproducibility