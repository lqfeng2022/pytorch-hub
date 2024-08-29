import { Box, Text } from '@chakra-ui/react'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import chapterOne from '../../data/chapterOne'

const ReproducibilitySection = () => {
  const [ 
    repro, rand, randpy, randf, pseudorand, pseudorandf, randseed, randseedpy
  ] = chapterOne[8].sections

  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>{repro.name}</Text>
        <Definition definition={repro}/>
      </Box>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>{rand.name}</Text>
        <Definition definition={rand}/>
      </Box>
      <BaseGrid section={randpy}/>
      <BaseGrid section={randf}/>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>{pseudorand.name}</Text>
        <Definition definition={pseudorand}/>
      </Box>
      <BaseGrid section={pseudorandf}/>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>{randseed.name}</Text>
        <Definition definition={randseed}/>
      </Box>
      <BaseGrid section={randseedpy}/>
    </>
  )
}

export default ReproducibilitySection